"""Fixed regression questions for the router and agent prompts.

Two modes:

* Offline (default, part of the normal test run): the case files are validated and
  the harness is exercised with scripted providers. Nothing leaves the process.
* Live (``REGRESSION=1 pytest tests/regression``): every case is sent to the real
  local router model and the real cloud model configured in ``.env``. Tools are
  replaced by stubs with canned output, so nothing is searched, fetched, executed
  or posted. Run this after changing a prompt or a tool description; it spends
  cloud tokens.
"""
