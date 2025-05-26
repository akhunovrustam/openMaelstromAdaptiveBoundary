#include <iostream>
#include <tools/log.h>


std::vector<std::tuple<log_level, std::chrono::system_clock::time_point, std::string>> logger::logs;
bool logger::silent = false;

std::function<void(std::string, log_level)> logger::log_fn = [](std::string message,
    log_level log) {
        std::stringstream os;
        //if (!silent) {
        switch (log) {
        case log_level::info:
            os << R"([info] 	)";
            break;
        case log_level::error:
            os << R"([error]	)";
            break;
        case log_level::debug:
            os << R"([debug]	)";
            break;
        case log_level::warning:
            os << R"([warning]	)";
            break;
        case log_level::verbose:
            os << R"([verbose]	)";
            break;
            default: break;
        }
        os << message << "\n";
        //std::cout << os.str();
      //}
        logs.push_back(std::make_tuple(log, std::chrono::system_clock::now(), message));
        //LogWindow::instance().AddLog(os.str().c_str());
};
#include <chrono>
#include <ctime>
#include <fstream>

void logger::write_log(std::string fileName) {
    std::ofstream file;
    file.open(fileName);

    for (auto log : logger::logs) {
        auto [level, time, message] = log;

        auto tt = std::chrono::system_clock::to_time_t(time);
        // Convert std::time_t to std::tm (popular extension)

        std::stringstream sstream;
        switch (level) {
        case log_level::info:
            sstream << R"([info])";
            break;
        case log_level::error:
            sstream << R"([error])";
            break;
        case log_level::debug:
            sstream << R"([debug])";
            break;
        case log_level::warning:
            sstream << R"([warning])";
            break;
        case log_level::verbose:
            sstream << R"([verbose])";
            break;
            default:break;
        }
        sstream << message;
        file << sstream.str() << std::endl;
    }
    file.close();
}
