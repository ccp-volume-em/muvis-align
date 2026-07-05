# Based on https://github.com/napari/napari/blob/main/dockerfile
# Base image with Python 3.11 (unable to build XPRA server with Python 3.12)
FROM python:3.11-slim-bookworm AS muvis-align

ARG DEBIAN_FRONTEND=noninteractive

# install python resources + graphical libraries used by qt and vispy
RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        git \
        libglib2.0-0 \
        mesa-utils \
        libglx-mesa0 \
        # tlambert03/setup-qt-libs
        libegl1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        x11-utils \
        libxcb-cursor0 \
        libopengl0 \
        # other/remaining
        libfontconfig1 \
        libxrender1 \
        libxi6 \
        libxcb-shape0 \
        && apt-get clean \

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY run.py .
COPY pyproject.toml .
COPY src/ src/

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install napari[all]
RUN pip install .

ENTRYPOINT ["python3", "-m", "napari", "--plugin", "muvis-align"]


FROM muvis-align AS muvis-align-xpra

ARG DEBIAN_FRONTEND=noninteractive

# Install Xpra and dependencies
# Remember to update the xpra.sources link for any change in distro version
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https \
    software-properties-common ca-certificates && \
    wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    wget -O "/etc/apt/sources.list.d/xpra.sources" https://raw.githubusercontent.com/Xpra-org/xpra/master/packaging/repos/bookworm/xpra.sources


RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
        menu-xdg \
        xdg-utils \
        xterm \
        sshfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_START="python3 -m napari --with muvis-align"
ENV XPRA_EXIT_WITH_CHILDREN="yes"
ENV XPRA_EXIT_WITH_CLIENT="no"
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
EXPOSE 9876

CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT or $(hostname -i):$XPRA_PORT"; \
    xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --start-child="$XPRA_START" \
    --exit-with-children="$XPRA_EXIT_WITH_CHILDREN" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY

ENTRYPOINT []

# Build:
# docker build -t muvis-align-xpra .
# Run:
# docker run -v "D:\slides:/data" -p 9876:9876 muvis-align-xpra
