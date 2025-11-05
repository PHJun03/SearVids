#include "whisper_wrapper.h"

#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <fstream>

#if defined(_WIN32)
    #define NOMINMAX
    #include <windows.h>
    #include <shellapi.h>
#else
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <signal.h>
    #include <errno.h>
#endif

namespace whisper_wrapper {

static std::string escape_arg_posix(const std::string& s) {
    // naive POSIX shell escaping for arguments passed as a single string to sh -c if needed.
    // But we build argv directly on POSIX, so not used there. Keep for safety.
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out += "'";
    return out;
}

WhisperWrapper::WhisperWrapper() {
    // default: try common executable names
    cliExe = "whisper"; // prefer 'whisper' if present
    argsTemplate = "{infile}"; // default just pass infile
}

WhisperWrapper::WhisperWrapper(std::string cli_executable_path)
    : cliExe(std::move(cli_executable_path)), argsTemplate("{infile}") {}

void WhisperWrapper::setCliExecutable(const std::string& path) {
    cliExe = path;
}

void WhisperWrapper::setCliArgsTemplate(const std::string& tpl) {
    argsTemplate = tpl;
}

bool WhisperWrapper::cli_available() const {
    // Very simple check: try opening the file (absolute path) or assume in PATH.
#if defined(_WIN32)
    DWORD attrs = GetFileAttributesA(cliExe.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY)) return true;
    // fallback: try where.exe
    std::string cmd = "where " + cliExe + " >nul 2>&1";
    int rc = std::system(cmd.c_str());
    return rc == 0;
#else
    // If cliExe contains '/', treat as path
    if (cliExe.find('/') != std::string::npos) {
        return (access(cliExe.c_str(), X_OK) == 0);
    }
    // else use 'which'
    std::string cmd = "which " + cliExe + " >/dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    return rc == 0;
#endif
}

std::vector<std::string> WhisperWrapper::buildCommand(const std::string& infile) const {
    std::vector<std::string> out;
    out.push_back(cliExe);

    // Simple parser: split argsTemplate by spaces, substitute {infile}
    std::istringstream iss(argsTemplate);
    std::string tok;
    while (iss >> tok) {
        size_t pos = tok.find("{infile}");
        if (pos != std::string::npos) {
            std::string replaced = tok;
            replaced.replace(pos, strlen("{infile}"), infile);
            out.push_back(replaced);
        } else {
            out.push_back(tok);
        }
    }
    // If argsTemplate was empty, ensure infile is appended
    if (out.size() == 1) out.push_back(infile);
    return out;
}

std::pair<int,std::string> WhisperWrapper::runCommandCapture(const std::vector<std::string>& argv, int timeout_seconds) const {
#if defined(_WIN32)
    // Build command line: quote each arg
    std::wstring cmdline;
    for (size_t i=0;i<argv.size();++i) {
        std::string a = argv[i];
        // Escape double quotes by doubling
        std::wstring wa;
        int needquotes = 0;
        for (char c : a) {
            if (c == ' ' || c == '\t') needquotes = 1;
        }
        // convert to wide string and quote
        int sl = MultiByteToWideChar(CP_UTF8, 0, a.c_str(), -1, nullptr, 0);
        std::wstring warg(sl, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, a.c_str(), -1, &warg[0], sl);
        if (needquotes) {
            cmdline += L"\"";
            // remove trailing \0
            if (!warg.empty() && warg.back() == L'\0') warg.pop_back();
            cmdline += warg;
            cmdline += L"\"";
        } else {
            if (!warg.empty() && warg.back() == L'\0') warg.pop_back();
            cmdline += warg;
        }
        if (i+1 < argv.size()) cmdline += L" ";
    }

    SECURITY_ATTRIBUTES saAttr{};
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    HANDLE hStdOutRead = NULL;
    HANDLE hStdOutWrite = NULL;
    if (!CreatePipe(&hStdOutRead, &hStdOutWrite, &saAttr, 0)) {
        return {-1, "CreatePipe failed"};
    }
    if (!SetHandleInformation(hStdOutRead, HANDLE_FLAG_INHERIT, 0)) {
        CloseHandle(hStdOutRead); CloseHandle(hStdOutWrite);
        return {-1, "SetHandleInformation failed"};
    }

    PROCESS_INFORMATION piProcInfo{};
    STARTUPINFOW siStartInfo{};
    siStartInfo.cb = sizeof(STARTUPINFOW);
    siStartInfo.hStdError = hStdOutWrite;
    siStartInfo.hStdOutput = hStdOutWrite;
    siStartInfo.hStdInput = NULL;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    // Create process
    BOOL ok = CreateProcessW(
        NULL,
        &cmdline[0], // command line (wchar_t*)
        NULL,
        NULL,
        TRUE,
        CREATE_NO_WINDOW,
        NULL,
        NULL,
        &siStartInfo,
        &piProcInfo
    );

    CloseHandle(hStdOutWrite);
    if (!ok) {
        CloseHandle(hStdOutRead);
        DWORD e = GetLastError();
        std::ostringstream oss;
        oss << "CreateProcess failed: " << e;
        return {-1, oss.str()};
    }

    std::string output;
    const DWORD bufSize = 4096;
    CHAR buffer[bufSize];
    DWORD read = 0;

    // Wait with timeout and read pipe while process runs
    DWORD waitMs = (timeout_seconds > 0) ? (DWORD)timeout_seconds * 1000 : INFINITE;
    DWORD waitResult;
    bool timedOut = false;
    for (;;) {
        // Read available data (non-blocking read using PeekNamedPipe)
        DWORD avail = 0;
        if (PeekNamedPipe(hStdOutRead, NULL, 0, NULL, &avail, NULL) && avail > 0) {
            if (ReadFile(hStdOutRead, buffer, std::min<DWORD>(bufSize-1, avail), &read, NULL) && read > 0) {
                buffer[read] = '\0';
                output.append(buffer, read);
            }
        }

        waitResult = WaitForSingleObject(piProcInfo.hProcess, 100); // check periodically
        if (waitResult == WAIT_OBJECT_0) {
            // process finished; drain remaining output
            while (PeekNamedPipe(hStdOutRead, NULL, 0, NULL, &avail, NULL) && avail > 0) {
                if (ReadFile(hStdOutRead, buffer, std::min<DWORD>(bufSize-1, avail), &read, NULL) && read > 0) {
                    buffer[read] = '\0';
                    output.append(buffer, read);
                } else break;
            }
            break;
        }

        // decrement waitMs
        if (timeout_seconds > 0) {
            if (waitMs <= 100) {
                // final timeout check
                timedOut = true;
                break;
            }
            waitMs = (waitMs > 100) ? (waitMs - 100) : 0;
        }
    }

    if (timedOut) {
        TerminateProcess(piProcInfo.hProcess, 1);
        CloseHandle(piProcInfo.hProcess);
        CloseHandle(piProcInfo.hThread);
        CloseHandle(hStdOutRead);
        return {-1, "timeout"};
    }

    DWORD exitCode = 0;
    GetExitCodeProcess(piProcInfo.hProcess, &exitCode);

    CloseHandle(piProcInfo.hProcess);
    CloseHandle(piProcInfo.hThread);
    CloseHandle(hStdOutRead);

    return { static_cast<int>(exitCode), output };

#else
    // POSIX implementation: fork/exec and pipe
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        return {-1, std::string("pipe failed: ") + std::strerror(errno)};
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        return {-1, std::string("fork failed: ") + std::strerror(errno)};
    }

    if (pid == 0) {
        // child
        // redirect stdout & stderr to pipefd[1]
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);

        // build argv for execvp
        std::vector<char*> cargs;
        for (const auto &s : argv) {
            cargs.push_back(const_cast<char*>(s.c_str()));
        }
        cargs.push_back(nullptr);
        // execute
        execvp(cargs[0], cargs.data());
        // if execvp fails:
        _exit(127);
    }

    // parent
    close(pipefd[1]);
    // set non-blocking read
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

    std::string output;
    const int bufSize = 4096;
    char buf[bufSize];
    bool finished = false;
    int status = 0;

    auto start = std::chrono::steady_clock::now();
    while (true) {
        // read available
        ssize_t r = read(pipefd[0], buf, bufSize-1);
        if (r > 0) {
            buf[r] = '\0';
            output.append(buf, r);
        }

        // check child status
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            finished = true;
            // drain remaining data
            while ((r = read(pipefd[0], buf, bufSize-1)) > 0) {
                buf[r] = '\0';
                output.append(buf, r);
            }
            break;
        }

        // timeout?
        if (timeout_seconds > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
            if (elapsed >= timeout_seconds) {
                // kill child
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                close(pipefd[0]);
                return {-1, "timeout"};
            }
        }

        // sleep a bit
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    close(pipefd[0]);
    int exit_code = -1;
    if (finished) {
        if (WIFEXITED(status)) exit_code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) exit_code = 128 + WTERMSIG(status);
    }
    return { exit_code, output };
#endif
}

std::string WhisperWrapper::transcribe_from_file(const std::string& infile, int timeout_seconds) const {
    if (cliExe.empty()) throw std::runtime_error("whisper CLI executable not configured");

    auto cmd = buildCommand(infile);
    auto [exit_code, output] = runCommandCapture(cmd, timeout_seconds);
    if (exit_code == -1) {
        // interpret "timeout" or other errors
        if (output == "timeout") throw std::runtime_error("Whisper transcription timed out");
        throw std::runtime_error(std::string("Whisper runner failed: ") + output);
    }
    if (exit_code != 0) {
        // return stderr+stdout in message
        throw std::runtime_error("Whisper process exited with code " + std::to_string(exit_code) + ". Output:\n" + output);
    }

    // Heuristic: CLI might produce metadata; try to extract plain transcript.
    // For simplicity, return whole stdout trimmed.
    // Caller can parse as needed (e.g., remove timestamps).
    // Trim trailing spaces/newlines:
    size_t s = 0;
    size_t e = output.size();
    while (s < e && (output[s] == '\n' || output[s] == '\r' || output[s] == ' ' || output[s] == '\t')) ++s;
    while (e > s && (output[e-1] == '\n' || output[e-1] == '\r' || output[e-1] == ' ' || output[e-1] == '\t')) --e;
    return output.substr(s, e - s);
}

} // namespace whisper_wrapper
