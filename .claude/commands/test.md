Run the test suite for the bn-en-translate project.

Steps:
1. Activate venv: `source .venv/bin/activate`
2. Run fast tests: `make test`
3. Report: number of passed/failed, any errors with their tracebacks.

If the user passes "slow" as argument, run `make test-slow` instead.
If the user passes "e2e", run `make test-e2e`.
If the user passes "all", run `make test-all`.
If any tests fail, diagnose the root cause before suggesting fixes.
