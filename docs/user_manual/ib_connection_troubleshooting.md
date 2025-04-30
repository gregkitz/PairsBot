# TWS API Connection Troubleshooting Guide

If you're experiencing issues connecting to Interactive Brokers TWS via the API, follow these troubleshooting steps to resolve the problem.

## Symptom: Connection Times Out After "Connected" Message

This is a common issue where the API client establishes an initial socket connection but doesn't complete the handshake. The log typically shows:

```
INFO - Connecting to localhost:7497 with clientId 999...
INFO - Connected
INFO - Disconnecting
ERROR - API connection failed: TimeoutError()
INFO - Disconnected.
```

## Solution Steps

### 1. Verify TWS Is Running and Logged In

- Make sure TWS is running and you're logged in
- Use Paper Trading for testing (port 7497)

### 2. Check API Configuration in TWS

1. Open TWS and go to **Edit → Global Configuration → API → Settings**
2. Verify these settings:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Socket port: 7497 (Paper Trading)
   - ✅ Allow connections from localhost only
   - ❌ Uncheck "Read-Only API"
   - ✅ Trust and allow "127.0.0.1" in Trusted IPs (if that setting exists)

3. Other helpful settings:
   - ✅ Create API message log file
   - ✅ Include market data in API message log
   - ✅ Let API account queries run in threads for better response time

4. Click **Apply** and **OK**

### 3. Check for API Authorization Dialogs

- TWS requires explicit permission for API connections
- When a script connects, TWS should display a popup asking for permission
- This dialog might be:
  - Hidden behind other windows
  - Minimized in the taskbar
  - In a different virtual desktop/space
  
- **Important:** If you can't find this dialog, try:
  - Minimizing all windows to look for it
  - Checking the TWS icon in the system tray for notification badges
  - Restarting TWS and trying again

### 4. Restart TWS and Clear Settings

If you still can't connect:

1. Close TWS completely
2. Delete TWS API settings cache:
   - On Windows: Delete files in `%USERPROFILE%\Jts` directory
   - On macOS: Delete files in `~/Jts` directory
3. Restart TWS
4. Reconfigure API settings as above
5. Try connecting again

### 5. Check TWS Logs

1. In TWS, go to **Help → Logs → API**
2. Look for error messages or rejected connection attempts
3. If you see "Authorization Error" messages, this confirms the security dialog is not being approved

### 6. Try a Different Client ID

- Each connection to TWS needs a unique client ID
- If another application is using client ID 1, try using 999 or another number
- In our scripts, we use client ID 999 to avoid conflicts

### 7. Adjust Connection Timeout

- Increase the connection timeout in the API client
- Our tests use a 60-second timeout which should be sufficient

### 8. Test with a Minimal Script

We've provided a debug script that tests just the connection:
```bash
./debug_ib_connection.py
```

This script isolates the connection process and includes a timeout to prevent hanging indefinitely.

### 9. Check TWS Version Compatibility

- Make sure your TWS version is compatible with the ib_insync library version
- TWS is regularly updated, and sometimes API libraries need updates too

## Additional Resources

- [IB Insync Documentation](https://ib-insync.readthedocs.io/)
- [TWS API Support Forum](https://groups.io/g/twsapi)
- [TWS User Guide - API Configuration](https://guides.interactivebrokers.com/tws/usersguidebook/configuretws/configure_tws.htm)

If you continue to experience issues, please contact Interactive Brokers support for assistance with their API configuration. 