# Sentinel Journal

## 2025-05-15 - [File Permissions and Error Masking]
**Vulnerability:** The application creates a state directory and log files with default system permissions, potentially exposing sensitive routing logs and health data to other users on the system. Additionally, internal provider error messages are returned directly to the client, which could leak sensitive information.
**Learning:** Default file creation in Python respects umask, which is often too permissive for sensitive data like AI conversation metadata and health metrics.
**Prevention:** Explicitly set directory permissions to 0700 and file permissions to 0600 when handling sensitive local state. Mask or sanitize error messages before returning them to external callers.
