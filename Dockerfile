# BstsNx showcase site — build context is the REPO ROOT because the site
# depends on the library via a path dependency ({:bsts_nx, path: ".."}).
# The layout inside the image mirrors the repo: /app is the library,
# /app/site is the Phoenix app.

ARG ELIXIR_VERSION=1.19.5
ARG OTP_VERSION=28.3.3
ARG DEBIAN_VERSION=trixie-20260623

ARG BUILDER_IMAGE="hexpm/elixir:${ELIXIR_VERSION}-erlang-${OTP_VERSION}-debian-${DEBIAN_VERSION}-slim"
ARG RUNNER_IMAGE="debian:${DEBIAN_VERSION}-slim"

FROM ${BUILDER_IMAGE} AS builder

# install build dependencies (git for the heroicons github dep)
RUN apt-get update -y && apt-get install -y build-essential git ca-certificates \
    && apt-get clean && rm -f /var/lib/apt/lists/*_*

WORKDIR /app

RUN mix local.hex --force && mix local.rebar --force

ENV MIX_ENV="prod"

# the library (path dependency of the site)
COPY mix.exs mix.lock ./
COPY lib lib

# the site
WORKDIR /app/site
COPY site/mix.exs site/mix.lock ./
RUN mix deps.get --only $MIX_ENV
RUN mkdir config

# compile-time config first for better layer caching
COPY site/config/config.exs site/config/prod.exs config/
RUN mix deps.compile

COPY site/priv priv
COPY site/lib lib
COPY site/assets assets

# tailwind + esbuild binaries download at build time, then digest
RUN mix assets.setup
RUN mix assets.deploy

RUN mix compile

COPY site/config/runtime.exs config/
COPY site/rel rel
RUN mix release

# ── runtime ──────────────────────────────────────────────────────────
FROM ${RUNNER_IMAGE}

RUN apt-get update -y && \
    apt-get install -y libstdc++6 openssl libncurses6 locales ca-certificates \
    && apt-get clean && rm -f /var/lib/apt/lists/*_*

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

WORKDIR "/app"
RUN chown nobody /app

ENV MIX_ENV="prod"

COPY --from=builder --chown=nobody:root /app/site/_build/${MIX_ENV}/rel/bsts_site ./

USER nobody

# Appended by flyctl
ENV ECTO_IPV6=true
ENV ERL_AFLAGS="-proto_dist inet6_tcp"

CMD ["/app/bin/server"]
