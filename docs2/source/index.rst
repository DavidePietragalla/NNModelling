NNModelling Documentation
=========================

**NNModelling** is a Domain-Specific Language (DSL) for designing neural
networks using a visual node editor. Diagrams are compiled to production-ready
PyTorch code via the Lightning framework and configured with Hydra.

The project consists of three main packages:

* **front-end/** — A Svelte 5 visual editor built with Svelte Flow (TypeScript)
* **converted/** — Python codegen target (PyTorch + Lightning + Hydra)
* **mcp-server/** — An MCP server that proxies diagram state from the browser
  to LLM agents

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   stereotypes
   python_api
   typescript_api
   examples
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
