@echo off
REM ============================================================
REM  FlowLLM - Build SUMO Network
REM  Compiles node, edge, connection, and traffic light XML files
REM  into a single .net.xml network file using netconvert.
REM ============================================================

echo [FlowLLM] Building intersection network...

netconvert ^
    --node-files=intersection.nod.xml ^
    --edge-files=intersection.edg.xml ^
    --connection-files=intersection.con.xml ^
    --tllogic-files=intersection.tll.xml ^
    --output-file=intersection.net.xml

if %ERRORLEVEL% equ 0 (
    echo [FlowLLM] Network built successfully: intersection.net.xml
) else (
    echo [FlowLLM] ERROR: netconvert failed. Make sure SUMO is installed and in PATH.
    exit /b 1
)
