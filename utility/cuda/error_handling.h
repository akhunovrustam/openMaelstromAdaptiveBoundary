#pragma once
#include <utility/cuda.h>
#include <utility/helpers.h>
#include <iostream>

enum struct error_level { last_error, device_synchronize, thread_synchronize };

#if defined(_WIN32) && !defined(__CUDA_ARCH__)
//#include <Psapi.h>
//#include <eh.h>
#include <windows.h>
#include <dbghelp.h>
#include <intrin.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <windows.h>
#pragma comment(lib, "dbghelp.lib")
#endif
namespace cuda {
#if defined(_WIN32) && !defined(__CUDA_ARCH__)

#define DBG_TRACE(MSG, ...) ::cuda::dbg::trace(MSG, __VA_ARGS__)

#define DBG_SOFT_ASSERT(COND)                                                                                          \
  if ((COND) == false) {                                                                                               \
    DBG_TRACE(__FUNCTION__ ": Assertion '" #COND "' failed!\n");                                                       \
  }

#define DBG_ASSERT(COND)                                                                                               \
  if ((COND) == false) {                                                                                               \
    DBG_TRACE(__FUNCTION__ ": Assertion '" #COND "' failed!\n");                                                       \
    ::cuda::dbg::handle_assert(__FUNCTION__, #COND);                                                                         \
  }

#define DBG_FAIL(MSG)                                                                                                  \
  DBG_TRACE(__FUNCTION__ MSG "\n");                                                                                    \
  ::cuda::dbg::fail(__FUNCTION__, MSG);

namespace dbg {
inline void trace(const char *msg, ...) {
  char buff[1024];

  va_list args;
  va_start(args, msg);
  vsnprintf(buff, 1024, msg, args);

  OutputDebugStringA(buff);

  va_end(args);
}

inline std::string basename(const std::string &file) {
  unsigned int i = (uint32_t)file.find_last_of("\\/");
  if (i == std::string::npos) {
    return file;
  } else {
    return file.substr(i + 1);
  }
}

struct StackFrame {
  DWORD64 address;
  std::string name;
  std::string module;
  unsigned int line;
  std::string file;
};

inline std::vector<StackFrame> stack_trace() {
#if _WIN64
  DWORD machine = IMAGE_FILE_MACHINE_AMD64;
#else
  DWORD machine = IMAGE_FILE_MACHINE_I386;
#endif
  HANDLE process = GetCurrentProcess();
  HANDLE thread = GetCurrentThread();

  if (SymInitialize(process, NULL, TRUE) == FALSE) {
    DBG_TRACE(__FUNCTION__ ": Failed to call SymInitialize.");
    return std::vector<StackFrame>();
  }

  SymSetOptions(SYMOPT_LOAD_LINES);

  CONTEXT context = {};
  context.ContextFlags = CONTEXT_FULL;
  RtlCaptureContext(&context);

#if _WIN64
  STACKFRAME frame = {};
  frame.AddrPC.Offset = context.Rip;
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Offset = context.Rbp;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Offset = context.Rsp;
  frame.AddrStack.Mode = AddrModeFlat;
#else
  STACKFRAME frame = {};
  frame.AddrPC.Offset = context.Eip;
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Offset = context.Ebp;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Offset = context.Esp;
  frame.AddrStack.Mode = AddrModeFlat;
#endif

  bool first = true;

  std::vector<StackFrame> frames;
  while (StackWalk(machine, process, thread, &frame, &context, NULL, SymFunctionTableAccess, SymGetModuleBase, NULL)) {
    StackFrame f = {};
    f.address = frame.AddrPC.Offset;

#if _WIN64
    DWORD64 moduleBase = 0;
#else
    DWORD moduleBase = 0;
#endif

    moduleBase = SymGetModuleBase(process, frame.AddrPC.Offset);

    char moduelBuff[MAX_PATH];
    if (moduleBase && GetModuleFileNameA((HINSTANCE)moduleBase, moduelBuff, MAX_PATH)) {
      f.module = basename(moduelBuff);
    } else {
      f.module = "Unknown Module";
    }
#if _WIN64
    DWORD64 offset = 0;
#else
    DWORD offset = 0;
#endif
    char symbolBuffer[sizeof(IMAGEHLP_SYMBOL) + 255];
    PIMAGEHLP_SYMBOL symbol = (PIMAGEHLP_SYMBOL)symbolBuffer;
    symbol->SizeOfStruct = (sizeof IMAGEHLP_SYMBOL) + 255;
    symbol->MaxNameLength = 254;

    if (SymGetSymFromAddr(process, frame.AddrPC.Offset, &offset, symbol)) {
      f.name = symbol->Name;
    } else {
      DWORD error = GetLastError();
      DBG_TRACE(__FUNCTION__ ": Failed to resolve address 0x%X: %u\n", frame.AddrPC.Offset, error);
      f.name = "Unknown Function";
    }

    IMAGEHLP_LINE line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE);

    DWORD offset_ln = 0;
    if (SymGetLineFromAddr(process, frame.AddrPC.Offset, &offset_ln, &line)) {
      f.file = line.FileName;
      f.line = line.LineNumber;
    } else {
      DWORD error = GetLastError();
      DBG_TRACE(__FUNCTION__ ": Failed to resolve line for 0x%X: %u\n", frame.AddrPC.Offset, error);
      f.line = 0;
    }

    if (!first) {
      frames.push_back(f);
    }
    first = false;
  }

  SymCleanup(process);

  return frames;
}

inline void handle_assert(const char *func, const char *cond) {
  std::stringstream buff;
  buff << func << ": Assertion '" << cond << "' failed! \n";
  buff << "\n";

  std::vector<StackFrame> stack = stack_trace();
  buff << "Callstack: \n";
  for (unsigned int i = 0; i < stack.size(); i++) {
    buff << "0x" << std::hex << stack[i].address << ": " << stack[i].name << "(" << std::dec << stack[i].line << ") in "
         << stack[i].module << "\n";
  }

  MessageBoxA(NULL, buff.str().c_str(), "Assert Failed", MB_OK | MB_ICONSTOP);
  abort();
}

inline void fail(const char *func, const char *msg) {
  std::stringstream buff;
  buff << func << ":  General Software Fault: '" << msg << "'! \n";
  buff << "\n";

  std::vector<StackFrame> stack = stack_trace();
  buff << "Callstack: \n";
  for (unsigned int i = 0; i < stack.size(); i++) {
    buff << "0x" << std::hex << stack[i].address << ": " << stack[i].name << "(" << stack[i].line << ") in "
         << stack[i].module << "\n";
  }

  MessageBoxA(NULL, buff.str().c_str(), "General Software Fault", MB_OK | MB_ICONSTOP);
  abort();
}
}
#endif


	template <typename T, typename U>
	void checkMessages(T errorMessage, U file = std::string(""), int32_t line = 0,
		error_level err_level = error_level::thread_synchronize) {
		auto err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::stringstream sstream;
			sstream << file << "@" << line << ": " << errorMessage << " ( " << cudaGetErrorString(err)
				<< " ) ";
			std::cerr << sstream.str() << std::endl;
			//logger(log_level::error) << sstream.str() << std::endl;
			throw std::runtime_error(sstream.str().c_str());
		}
		if (err_level == error_level::device_synchronize ||
			err_level == error_level::thread_synchronize) {
			auto err = cuda::sync_quiet();
			if (err != cudaSuccess) {
				std::stringstream sstream;
				sstream << file << "@" << line << ": " << errorMessage << " ( " << cudaGetErrorString(err)
					<< " ) ";
				std::cerr << sstream.str() << std::endl;
#if defined(_WIN32) && !defined(__CUDA_ARCH__)

                                auto trace = cuda::dbg::stack_trace();
                                for (auto t : trace) {
                                  std::cerr << t.address << ": " << t.file << " - " << t.line << " => " << t.name
                                            << " @ " << t.module << std::endl;
                                }
                #endif
				//logger(log_level::error) << sstream.str() << std::endl;
				throw std::runtime_error(sstream.str().c_str());
			}
		}
		//if (err_level == error_level::thread_synchronize) {
		//	auto err = cudaThreadSynchronize();
		//	if (err != cudaSuccess) {
		//		std::stringstream sstream;
		//		sstream << file << "@" << line << ": " << errorMessage << " ( " << cudaGetErrorString(err)
		//			<< " ) ";
		//		std::cerr << sstream.str() << std::endl;
		//		//logger(log_level::error) << sstream.str() << std::endl;
		//		throw std::runtime_error(sstream.str().c_str());
		//	}
		//}
	}

	cudaError_t error_check(std::string message = "");

	cudaError_t error_check(cudaError_t err, std::string message = "");

	cudaError_t error_check_quiet(std::string message = "");

	cudaError_t error_check_quiet(cudaError_t err, std::string message = "");

}

#define CHECK_CUDA(x) cuda::checkMessages(x, __FILE__, __LINE__);
#define CHECK_ERROR cuda::checkMessages("", __FILE__, __LINE__);
