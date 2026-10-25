# Security boundaries

The installer is a local command run by the account that owns the editor
configuration. Its directory flags intentionally select arbitrary destinations
that account can write; they are not paths received from a remote client or a
model tool call. Do not expose this installer as a privileged service.

Before copying, it checks the entire plan for source and destination links,
source overlap and incompatible filesystem entries. It creates private temporary
files and atomically replaces destination entries, preserving unrelated files
and avoiding hard-link truncation. Installed files retain owner-only POSIX
permissions. Windows access control depends on the parent directory ACL.
Destination ancestors must be controlled by the installing account; these checks
do not isolate against another process concurrently renaming those ancestors.

CodeQL path-injection findings on the installer's explicit directory arguments
represent the intended local file-copy capability. Review that trust boundary
when changing callers. This rationale does not apply to paths from a network
request, an untrusted archive or a model response.

The Claude authorization test reads only the checked-in workflow condition and
uses a limited expression parser that cannot execute JavaScript. Unsupported
syntax fails the test. It does not accept an environment-selected source file.
CI executes repository tests without repository secrets or write access.
