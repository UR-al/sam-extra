# Vendored from https://github.com/eduardoabreu81/forgeneo-mcp at commit a103dc5
# (forgeneo_mcp/__init__.py) into sam-extra (mcp_server/sam_extra_mcp/forgeneo/__init__.py).
#
# Copyright (c) 2026 Eduardo Abreu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes in sam-extra (2026-10-03):
# - Package renamed from forgeneo_mcp to sam_extra_mcp.forgeneo; the upstream commit is recorded below.
# - Docstring notes that nothing in this package imports the MCP SDK.

"""MCP bridge between an AI agent and Stable Diffusion WebUI Forge Neo.

The vendored core of forgeneo-mcp. Nothing in this package imports the MCP SDK
(``sam_extra_mcp.server`` does), and only ``client`` and ``fetcher`` need httpx at
call time, so the logic is testable in an environment that has neither.
"""

__version__ = "0.1.0"
UPSTREAM_REPOSITORY = "https://github.com/eduardoabreu81/forgeneo-mcp"
UPSTREAM_COMMIT = "a103dc5"
