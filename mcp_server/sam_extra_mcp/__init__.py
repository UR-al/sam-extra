"""sam-extra MCP server: lets an MCP client on this PC (Claude Code and others) drive this Forge.

It runs in its own uv environment, never inside Forge: the MCP SDK needs pydantic>=2.11 while
Forge pins 2.10.6 and re-installs its pins at every launch. Register it once, user scope:

    claude mcp add --scope user sam-extra -- uv run --project <extension>/mcp_server sam-extra-mcp

What the agent may change (generate, switch checkpoints, interrupt, download modules) is set in
Forge -> Settings -> SAM Extra MCP and enforced by this server (``policy``).

Only ``server`` imports the MCP SDK; ``service``, ``policy``, ``forge_paths`` and the vendored
``forgeneo`` core (forgeneo-mcp, MIT, Copyright (c) 2026 Eduardo Abreu) do not.
"""

__version__ = "0.1.0"
