# Simplifying user access

## Local

No script: once published, install muvis-align in napari from the napari hub (or `pip install muvis-align`).

## HPC from the user's machine

`xpra-connect.sh` (bash, Git Bash on Windows) runs `xpra-run.sh` on the login node, tunnels to the compute node
and opens the browser logged in (see docs/technical.md). It assumes an ssh key on the login node: it opens up to
three ssh connections (session, tunnel, cancel), and Git Bash's ssh - like Windows' own OpenSSH - cannot share one
connection between them (ControlMaster), so without a key each one asks for the password.

Options to ask for the password once or never, if keys turn out to be a hurdle:

1. **Offer to set up a key on first use** (preferred if the cluster accepts keys without a second factor): check
   with `ssh -o BatchMode=yes <destination> true`; if that fails, offer to create a key and append it to
   `~/.ssh/authorized_keys` on the login node - one password prompt, ever. No change to the session flow.
2. **Ask once, hand it to each ssh** via an `SSH_ASKPASS` helper (`SSH_ASKPASS_REQUIRE=force`, OpenSSH 8.4+).
   Fails with 2FA/one-time codes; the script holds the password for the session; a first connection needs host
   keys accepted without asking (`StrictHostKeyChecking=accept-new`).
3. **One ssh connection for everything**: forward the local port to a unix socket on the login node
   (`-L <port>:<socket path>`, connected only when the browser connects), and relay that socket to the compute
   node once the job reports it (needs python3 or socat on the login node). Cancel goes over the same connection,
   e.g. a `cancel` line on its input. Works with 2FA, one prompt a session; the most code.

Open: whether the cluster login nodes accept keys without 2FA.
