#!/usr/bin/env bash
#
# Starts a muvis-align session on the cluster from your own machine and opens it in your browser:
# copies xpra-run.sh and xpra-pull.sh from beside it to the cluster, runs xpra-run.sh on the login node,
# tunnels to the compute node it starts on, and logs in.
#
# Usage, on your own machine (on Windows in Git Bash, or .\xpra-connect.sh from PowerShell, in a new window):
#   bash xpra-connect.sh <ssh destination> [folder] [--pull]
#     ssh destination  what you ssh into: user@login-node, or a Host from your ~/.ssh/config
#     folder           where to keep the scripts and container on the cluster, from your home or absolute
#                      (default: muvis-align); the scripts there are replaced each run, so edit them here
#     --pull           refresh the container first (xpra-pull.sh); done anyway when there is none yet
#   MUVIS_PARTITION, MUVIS_DATA_DIR, MUVIS_CONTAINER_DIR and MUVIS_LOGIN_NODE set here are passed on.
# With an ssh key you are not asked for a password; without one, up to three times.

set -euo pipefail

TUNNEL_PID=""
stop_tunnel() {
    if [ -n "${TUNNEL_PID}" ]; then
        kill "${TUNNEL_PID}" 2>/dev/null || true
        TUNNEL_PID=""
    fi
}
# started from PowerShell or Explorer, Git Bash runs it in a window of its own that closes as it ends
OWN_WINDOW=""
if ps -p "${PPID}" 2>/dev/null | grep -q mintty; then
    OWN_WINDOW=1
    # it starts as a login shell, whose logout (Git Bash's bash_logout) clears the window before it can be read
    if shopt -q login_shell; then
        exec bash "$0" "$@"
    fi
fi
on_exit() {
    stop_tunnel
    if [ -n "${OWN_WINDOW}" ]; then
        read -r -p "Press Enter to close this window." IGNORED || true
    fi
}
trap on_exit EXIT

PULL=""
ARGUMENTS=()
for ARGUMENT in "$@"; do
    if [ "${ARGUMENT}" = "--pull" ]; then
        PULL=1
    else
        ARGUMENTS+=("${ARGUMENT}")
    fi
done
if [ "${#ARGUMENTS[@]}" -lt 1 ] || [ "${#ARGUMENTS[@]}" -gt 2 ]; then
    echo "Usage: bash xpra-connect.sh <ssh destination> [folder on the cluster, default: muvis-align] [--pull]" >&2
    exit 1
fi
DESTINATION="${ARGUMENTS[0]}"
REMOTE_DIR="${ARGUMENTS[1]:-muvis-align}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

SSH=(ssh)
CURL=curl
# WSL has a home of its own: without your ssh setup there, Windows' ssh has it
if grep -qi microsoft /proc/version 2>/dev/null && command -v ssh.exe > /dev/null 2>&1 \
        && ! ssh -o BatchMode=yes -o ConnectTimeout=15 "${DESTINATION}" true 2>/dev/null; then
    echo "WSL's ssh cannot sign in to ${DESTINATION}: using Windows' ssh."
    SSH=(ssh.exe)
    # its tunnel listens on Windows' localhost, which WSL's own tools may not reach
    CURL=curl.exe
fi
SSH+=(-o ServerAliveInterval=30)
case "${SSH[0]}:$(uname -s)" in
    ssh.exe:*|*:MINGW*|*:MSYS*|*:CYGWIN*)
        # Windows' and Git Bash's ssh cannot share a connection: each ssh signs in again
        ;;
    *)
        mkdir -p -m 700 "${HOME}/.ssh"
        SSH+=(-o ControlMaster=auto -o "ControlPath=${HOME}/.ssh/muvis-align-%C" -o ControlPersist=60)
        ;;
esac

# runs in a login shell on the cluster, read from its input: the commands in it get /dev/null instead
REMOTE_SCRIPT="set -e
mkdir -p $(printf %q "${REMOTE_DIR}")
cd $(printf %q "${REMOTE_DIR}")
"
# the cluster runs the scripts beside this one, so the two sides always match
for SCRIPT in xpra-run.sh xpra-pull.sh; do
    REMOTE_SCRIPT+="cat > ${SCRIPT}.new <<'MUVIS_SCRIPT_END'
$(tr -d '\r' < "${LOCAL_DIR}/${SCRIPT}")
MUVIS_SCRIPT_END
mv -f ${SCRIPT}.new ${SCRIPT}
"
done
REMOTE_SCRIPT+="export MUVIS_FOLLOW=1"
for NAME in MUVIS_PARTITION MUVIS_DATA_DIR MUVIS_CONTAINER_DIR MUVIS_LOGIN_NODE; do
    if [ -n "${!NAME+set}" ]; then
        REMOTE_SCRIPT+=" ${NAME}=$(printf %q "${!NAME}")"
    fi
done
if [ -n "${PULL}" ]; then
    REMOTE_SCRIPT+="
bash xpra-pull.sh < /dev/null"
else
    REMOTE_SCRIPT+='
[ -e "${MUVIS_CONTAINER_DIR:-.}/muvis-align-xpra_latest" ] || bash xpra-pull.sh < /dev/null'
fi
REMOTE_SCRIPT+="
bash xpra-run.sh < /dev/null"

JOB_ID=""
on_interrupt() {
    stop_tunnel
    echo
    if [ -n "${JOB_ID}" ]; then
        read -r -p "Stop job ${JOB_ID} on the cluster as well? [Y/n] " ANSWER || ANSWER="n"
        case "${ANSWER}" in
            n|N|no|No)
                echo "Job ${JOB_ID} keeps going; cancel it later with: ssh ${DESTINATION} scancel ${JOB_ID}"
                ;;
            *)
                "${SSH[@]}" "${DESTINATION}" scancel "${JOB_ID}" && echo "Job ${JOB_ID} cancelled."
                ;;
        esac
    fi
    exit 130
}
trap on_interrupt INT

echo "Starting a session through ${DESTINATION} in ${REMOTE_DIR} (Ctrl-C to stop) ..."
exec 3< <("${SSH[@]}" "${DESTINATION}" 'bash -l -s' <<< "${REMOTE_SCRIPT}")

SESSION=""
while [ -z "${SESSION}" ] && IFS= read -r LINE <&3; do
    case "${LINE}" in
        "MUVIS_SESSION "*)
            SESSION="${LINE}"
            ;;
        "Submitted job "*)
            JOB_ID="${LINE#Submitted job }"
            JOB_ID="${JOB_ID%%.*}"
            echo "${LINE}"
            ;;
        *)
            echo "${LINE}"
            ;;
    esac
done
if [ -z "${SESSION}" ]; then
    echo "No session was started (see above)." >&2
    exit 1
fi

for FIELD in ${SESSION#MUVIS_SESSION }; do
    case "${FIELD}" in
        job=*) JOB_ID="${FIELD#job=}" ;;
        node=*) NODE="${FIELD#node=}" ;;
        port=*) REMOTE_PORT="${FIELD#port=}" ;;
        user=*) REMOTE_USER="${FIELD#user=}" ;;
        password=*) PASSWORD="${FIELD#password=}" ;;
    esac
done

LOCAL_PORT=9876
# a port in use answers; another session from this machine may hold it
while (exec 4<>"/dev/tcp/127.0.0.1/${LOCAL_PORT}") 2>/dev/null; do
    LOCAL_PORT=$((LOCAL_PORT + 1))
done
start_tunnel() {
    "${SSH[@]}" -N -o ExitOnForwardFailure=yes -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "${DESTINATION}" &
    TUNNEL_PID=$!
}
echo "Job ${JOB_ID} runs on ${NODE}: waiting for napari ..."
start_tunnel
# the session is announced before xpra listens; Windows' ports are out of sight of the check above from WSL
SECONDS=0
until "${CURL}" -s "http://localhost:${LOCAL_PORT}/" | grep -i xpra > /dev/null; do
    if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
        TUNNEL_PID=""
        if [ "${LOCAL_PORT}" -ge 9895 ]; then
            echo "No tunnel could be opened (see above). Job ${JOB_ID} keeps going; Ctrl-C to stop it." >&2
            break
        fi
        # most likely another program on this machine has the port
        LOCAL_PORT=$((LOCAL_PORT + 1))
        start_tunnel
    elif [ "${SECONDS}" -ge 300 ]; then
        echo "napari did not answer within 5 minutes; the server log is ~/.xpra/job-${JOB_ID}/xpra.log on the cluster." >&2
        break
    fi
    sleep 2
done

if [ -n "${TUNNEL_PID}" ]; then
    # xpra's page logs in with these, and only sends a password from the URL to localhost
    URL="http://localhost:${LOCAL_PORT}/?username=${REMOTE_USER}&password=${PASSWORD}"
    if command -v rundll32.exe > /dev/null 2>&1; then
        rundll32.exe url.dll,FileProtocolHandler "${URL}"
    elif [ "$(uname -s)" = "Darwin" ]; then
        open "${URL}"
    else
        xdg-open "${URL}" > /dev/null 2>&1 || true
    fi
    echo "napari is at http://localhost:${LOCAL_PORT}/ (user ${REMOTE_USER}, password ${PASSWORD})."
    echo "Close napari to end the session, or Ctrl-C here to stop it."
fi

# the remote side ends its output when the job ends
while IFS= read -r LINE <&3; do
    echo "${LINE}"
done
stop_tunnel
