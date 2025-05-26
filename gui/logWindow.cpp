#include "glui.h"
#include <tools/log.h>

struct LoggingWindow {
  ImGuiTextBuffer Buf;
  ImGuiTextFilter Filter;
  ImVector<int> LineOffsets;
  bool AutoScroll;
  LoggingWindow() {
    AutoScroll = true;
    Clear();
  }
  void Clear() {
    Buf.clear();
    LineOffsets.clear();
    LineOffsets.push_back(0);
  }
  void AddLog(const char *fmt, ...) IM_FMTARGS(2) {
    int old_size = Buf.size();
    va_list args;
    va_start(args, fmt);
    Buf.appendfv(fmt, args);
    va_end(args);
    for (int new_size = Buf.size(); old_size < new_size; old_size++)
      if (Buf[old_size] == '\n')
        LineOffsets.push_back(old_size + 1);
  }
  void Draw(const char *title, bool *p_open = NULL) {
    {
      static std::size_t logSize = 0;
      auto logs = logger::logs;
      if (logs.size() != logSize) {
        for (int32_t i = logSize; i < logs.size(); ++i) {
          auto [log, time, message] = logs[i];
          std::stringstream os;
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
          }
          os << message;
          AddLog(os.str().c_str());
        }
      }
      logSize = logs.size();
    }
    if (!ImGui::Begin(title, p_open)) {
      ImGui::End();
      return;
    }
    // Options menu
    if (ImGui::BeginPopup("Options")) {
      ImGui::Checkbox("Auto-scroll", &AutoScroll);
      ImGui::EndPopup();
    }
    // Main window
    if (ImGui::Button("Options"))
      ImGui::OpenPopup("Options");
    ImGui::SameLine();
    bool clear = ImGui::Button("Clear");
    ImGui::SameLine();
    bool copy = ImGui::Button("Copy");
    ImGui::SameLine();
    Filter.Draw("Filter", -100.0f);
    ImGui::Separator();
    ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
    if (clear)
      Clear();
    if (copy)
      ImGui::LogToClipboard();
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    const char *buf = Buf.begin();
    const char *buf_end = Buf.end();
    if (Filter.IsActive()) {
      for (int line_no = 0; line_no < LineOffsets.Size; line_no++) {
        const char *line_start = buf + LineOffsets[line_no];
        const char *line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;
        if (Filter.PassFilter(line_start, line_end))
          ImGui::TextUnformatted(line_start, line_end);
      }
    } else {
      ImGuiListClipper clipper;
      clipper.Begin(LineOffsets.Size);
      while (clipper.Step()) {
        for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++) {
          const char *line_start = buf + LineOffsets[line_no];
          const char *line_end = (line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;
          ImGui::TextUnformatted(line_start, line_end);
        }
      }
      clipper.End();
    }
    ImGui::PopStyleVar();
    // ImGui::PopFont();
    if (AutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
      ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
    ImGui::End();
  }

  static LoggingWindow &instance() {
    static LoggingWindow inst;
    return inst;
  }
};
void GUI::LogWindow(bool* p_open){
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    LoggingWindow::instance().Draw("Log", p_open);
}
