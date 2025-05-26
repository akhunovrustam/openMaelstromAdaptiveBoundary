#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/type_traits/is_assignable.hpp>
#include <boost/type_traits/is_volatile.hpp>
#include <fstream>
#include <iostream>
#include <functional>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using node_t = std::pair<const std::string, boost::property_tree::ptree>;

std::map<std::string, std::vector<std::string>> global_map;
std::vector<std::string> complex_types;

struct transformation {
  std::vector<std::shared_ptr<transformation>> children;
  transformation *parent = nullptr;

  node_t *node;

  std::vector<std::tuple<std::string, std::string, std::string>> source;
  std::vector<std::tuple<std::string, std::string, std::string>> header;

  std::function<void(transformation &, transformation *, node_t &)> transfer_fn = [](auto &node, auto, node_t &tree) {
    for (auto &t_fn : node.children)
      for (auto &children : tree.second)
        t_fn->transform(children);
  };

  transformation(transformation *p = nullptr) : parent(p) {}
  transformation(decltype(transfer_fn) t_fn, transformation *p = nullptr) : parent(p), transfer_fn(t_fn) {}

  std::shared_ptr<transformation> addChild() {
    children.push_back(std::make_shared<transformation>(transfer_fn, this));
    return children[children.size() - 1];
  }

  std::shared_ptr<transformation> addChild(decltype(transfer_fn) t_fn) {
    children.push_back(std::make_shared<transformation>(t_fn, this));
    return children[children.size() - 1];
  }

  void transform(node_t &t_node) {
    node = &t_node;
    transfer_fn(*this, parent, t_node);
  }
  std::string tuple_h() {
	  std::stringstream sstream;
	  std::sort(header.begin(), header.end(), [](const auto& lhs, const auto& rhs) {return std::get<0>(lhs) < std::get<0>(rhs); });
	  std::string ns = "";
	  for (auto&[nameSpace, identifier, text] : header) {
		  if (ns != nameSpace && ns != "")
			  sstream << "}\n";
		  if (ns != nameSpace) {
			  sstream << "namespace " << nameSpace << "{\n";
			  ns = nameSpace;
		  }
		  auto id = identifier;
		  auto ambiF = std::count_if(header.begin(), header.end(), [=](auto&& val) {
			  return std::get<1>(val) == id;
		  });
		  auto textCopy = text;
		  if (ambiF != 1) {
			  identifier[0] = ::toupper(identifier[0]);
			  textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi_identifier)"), nameSpace + identifier);
			  textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi)"), "true");
		  }
		  else {
			  textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi_identifier)"), identifier);
			  textCopy = std::regex_replace(textCopy, std::regex(R"(\$ambi)"), "false");

		  }
		  sstream << textCopy;
	  }
	  if (header.size() != 0)
		  sstream << "}\n";
	  //sstream << boost::join(header,"\n") << std::endl;
	  for (auto& child : children)
		  sstream << child->tuple_h();
	  return sstream.str();
  }
  std::string tuple_s() {
	  std::stringstream sstream;
	  std::sort(source.begin(), source.end(), [](const auto& lhs, const auto& rhs) {return std::get<0>(lhs) < std::get<0>(rhs); });
	  for (auto&[nameSpace, identifier, text] : source) {
		  sstream << text;
	  }
	  //sstream << boost::join(source, "\n");
	  for (auto& child : children)
		  sstream << child->tuple_s();
	  return sstream.str();
  }
};

#if defined(_MSC_VER) && !defined(__CLANG__)
#define template_flag
#else
#define template_flag template
#endif
#define TYPE node.second.template_flag get<std::string>("type")
#define NAME parent->node->first

#include <filesystem>
namespace fs = std::filesystem;
#include <config/config.h>
fs::file_time_type newestFile;

auto resolveIncludes(boost::property_tree::ptree &pt) {
  fs::path source(sourceDirectory);
  auto folder = source / "jsonFiles";
  for (auto &fileName : fs::directory_iterator(folder)) {
    auto t = fs::last_write_time(fileName);
    if (t > newestFile)
      newestFile = t;
    std::stringstream ss;
    std::ifstream file(fileName.path().string());
    ss << file.rdbuf();
    file.close();

    boost::property_tree::ptree pt2;
    boost::property_tree::read_json(ss, pt2);
    auto arrs = pt2.get_child_optional("uniforms");
    if (arrs) {
      for (auto it : arrs.get()) {
        // std::cout << it.first << " : " << it.second.data() << std::endl;
        pt.add_child(it.first, it.second);
      }
    }
  }
  auto arrays = fs::path(sourceDirectory) / "parameters.json";
  // if(fs::exists(arrays) && fs::last_write_time(arrays) > newestFile)
  // 	return;
  // std::cout << "Writing " << arrays << std::endl;
  // std::ofstream file(arrays.string());
  // boost::property_tree::write_json(file, pt);
}
std::vector<std::string> namespaces;

int main(int argc, char **argv) try {
	std::cout << "Running uniform meta-code generation" << std::endl;
	return 0;
	// 
	//if (argc == 2) // {
	//  std::cerr << "Wrong number of arguments provided" << std::endl;
	//  for (int32_t i = 0; i < argc; ++i) {
	//	  std::cout << i << ": " << argv[i] << std::endl;
	//  }
	//  return 1;
 // }
  fs::path output(R"(C:/dev/source/MaelstromDev-original/utility/identifiers/uniform)");
  if(argc > 1)
    output = argv[1];

  std::stringstream ss;

  boost::property_tree::ptree pt;
  resolveIncludes(pt);
  // boost::property_tree::read_json(ss, pt);



  std::stringstream parameterCode;
  std::stringstream functionCode;
  std::stringstream functionCodeHost;
  std::stringstream functionCodeCU;
  std::stringstream functionCodeCU2;
  std::stringstream parameterCode2;
  std::stringstream structCode;

  std::map<std::string, std::vector<std::pair<std::string, std::string>>> enumIdentifiers;
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> enumUIdentifiers;
  std::map<std::string, std::vector<std::pair<std::string, boost::property_tree::ptree>>> nsNodes;
  std::vector<boost::property_tree::ptree> complexTypes;
  for (auto ns : pt)
	  for (auto uniform : ns.second)
		  nsNodes[ns.first].push_back(uniform);
  for (auto ns : nsNodes) {
	  //std::cout << ns.first << ":" << std::endl;
	  for (auto uniform : ns.second) {
		  //if (uniform.first != "volume$") continue;
		  auto node = uniform.second;

		  auto nameSpace = ns.first;
		  auto identifier = uniform.first.find("$") == std::string::npos ? uniform.first : uniform.first.substr(0, uniform.first.find("$"));
		  auto description = node.get<std::string>("name", "");
		  auto type = node.get<std::string>("type");
		  auto constant = node.get("const", true);
		  auto hidden = node.get("visible", false);
		  auto def = node.get<std::string>("default");
		  auto range = node.get_child_optional("range");
		  auto presets = node.get<std::string>("presets", "");
		  auto unit = node.get<std::string>("unit", "none");

		  if (type == "std::string" && presets.find("[]") == std::string::npos)
			  if (presets != "") {
				  std::vector<std::string> presetsVector;
				  std::stringstream ss(presets);
				  while (ss.good()) {
					  std::string substr;
					  std::getline(ss, substr, ',');
					  presetsVector.push_back(substr);
				  }
				  presets = "";
				  for (int32_t i = 0; i < presetsVector.size(); ++i) {
					  presets += (i == 0 ? std::string("") : std::string(",")) + std::string("std::string(\"") + presetsVector[i] + std::string("\")");
				  }
			  }

		  //struct ParameterSettings {
		  //	std::string description = "";
		  //	bool constant = false;
		  //	bool hidden = false;
		  //	std::vector<std::type_index> alternativeTypes{};
		  //	std::optional<Range> range;
		  //	std::vector<detail::iAny> presets{};
		  //};
		  auto type2 = type;
		  type2[0] = std::toupper(type2[0]);
if(presets != ""){
		  if (type == "std::string" && presets.find("[]") == std::string::npos)
			parameterCode << "static std::vector<" << type << "> " << nameSpace << "_" << identifier << "_presets = {" << presets << "};\n";
			else parameterCode << "static std::vector<" << type << "> " << nameSpace << "_" << identifier << "_presets = " << presets << ";\n";
}
if (range) {
	parameterCode << "static " << "auto" << " " << nameSpace << "_" << identifier << "_min = " << range.value().get<std::string>("min") << ";\n";
	parameterCode << "static " << "auto" << " " << nameSpace << "_" << identifier << "_max = " << range.value().get<std::string>("max") << ";\n";
}
		  parameterCode << "ParameterManager::instance().newParameter(\""
			  << nameSpace << "." << identifier << "\", " << type << "{" << (type == "std::string" ? "\"" : "") << def << (type == "std::string" ? "\"" : "") << "},{\n"
			  << "\t\t.description = \"" << description << "\",\n"
			  << "\t\t.constant = " << (constant ? "true" : "false") << ",\n"
			  << "\t\t.hidden = " << (hidden ? "true" : "false") << ",\n"
			  //<< "\t\t.alternativeTypes{},\n"
			  << "\t\t.alternativeTypes{" << (unit == "none" ? "" : "typeid(u" + type2 + "<" + unit + ">)") << "},\n"
			  //<< (!range ? "" : "\t\t.range = Range{" + range.value().get<std::string>("min") + ", " + range.value().get<std::string>("max") + "},\n");
			  << (!range ? "" : "\t\t.range = Range{" + 
				  nameSpace + "_" + identifier + "_min" + ", " +
				  nameSpace + "_" + identifier + "_max" + "},\n");
			  if (presets != "")
			  	parameterCode << "\t\t.presets = " << nameSpace << "_" << identifier << "_presets\n";
			  parameterCode  << "\t});\n";

		  //pm.newParameter("custom.type", customType{ 1.f,456,{2,6,10},std::string("something") }, { .description = "Test custom type" });
		  //std::cout << "\tNamespace: " << ns.first << std::endl;
		  //std::cout << "\t\tIdentifier: " << uniform.first << std::endl;
		  //std::cout << "\t\t\tDescription: " << node.get<std::string>("name", "") << std::endl;
		  //std::cout << "\t\t\tType: " << node.get<std::string>("type") << std::endl;
		  //std::cout << "\t\t\tconstant: " << node.get("const", true) << std::endl;
		  //std::cout << "\t\t\thidden: " << node.get("visible", false) << std::endl;
		  //std::cout << "\t\t\tdefault: " << node.get<std::string>("default") << std::endl;
		  //auto range = node.get_child_optional("range");
		  //if (range)
		  //	std::cout << "\t\t\trange " << range.value().get<std::string>("min") << " : " << range.value().get<std::string>("max") << std::endl;
		  //std::cout << "\t\t\tpresets: " << node.get<std::string>("presets", "") << std::endl;
		  //std::cout << "\t\t\tunit: " << node.get<std::string>("unit", "none") << std::endl;
		  auto complex = node.get_child_optional("complex_type");
		  if (complex) {
			  auto complexNode = complex.value();
			  complexTypes.push_back(complexNode);
		  }

		  enumIdentifiers[ns.first].push_back(std::make_pair(uniform.first.find("$") != std::string::npos ? uniform.first.substr(0, uniform.first.find("$")) : uniform.first, node.get<std::string>("type")));
		  enumUIdentifiers[ns.first].push_back(std::make_pair(uniform.first.find("$") != std::string::npos ? uniform.first.substr(0, uniform.first.find("$")) : uniform.first, (unit == "none" ? node.get<std::string>("type") : "u" + type2 + "<" + unit + ">")));
	  }
  }

  functionCode << "namespace parameters{\n";
  for (auto enumID : enumIdentifiers) {
	  functionCode << "\tenum struct " << enumID.first << " : std::size_t {";
	  for (int32_t i = 0; i < enumID.second.size() - 1; ++i)
		  functionCode << enumID.second[i].first << " = " << std::hash<std::string>{}("parameters::" + enumID.first + "::" + enumID.second[i].first) << "ull, ";
	  functionCode << enumID.second.rbegin()->first << " = " << std::hash<std::string>{}("parameters::" + enumID.first + "::" + enumID.second.rbegin()->first) << "ull";
	  functionCode << "};\n";
  }
  functionCodeHost << "\ttemplate<typename Func, typename... Ts>\n";
  functionCodeHost << "\tvoid iterateParameters(Func&& fn, Ts&&... args){\n";
  for (auto enumID : enumIdentifiers) {
	  for (auto e : enumID.second)
		  functionCodeHost << "\t\tfn(get<" << enumID.first << "::" << e.first << ">(), \"" << enumID.first << "." << e.first << "\", args...);\n";
  }
  functionCodeHost << "\t}\n";
  functionCodeHost << "\ttemplate<typename Func, typename... Ts>\n";
  functionCodeHost << "\tvoid iterateParametersU(Func&& fn, Ts&&... args){\n";
  for (auto enumID : enumUIdentifiers) {
	  for (auto e : enumID.second)
		  functionCodeHost << "\t\tfn(uGet<" << enumID.first << "::" << e.first << ">(), \"" << enumID.first << "." << e.first << "\", args...);\n";
  }
  functionCodeHost << "\t}\n";
  functionCodeHost << "}\n";
  //for (auto enumID : enumIdentifiers) {
	 // functionCodeHost << "template<parameters::" << enumID.first << " ident> auto& get(parameters::" << enumID.first << " = ident){\n";
	 // for (auto e : enumID.second)
		//  functionCodeHost << "\tif constexpr(ident == parameters::" << enumID.first << "::" << e.first << ") return ParameterManager::instance().get<" << e.second << ">(\"" << enumID.first << "." << e.first << "\");\n";
	 // functionCodeHost << "};\n";
  //}
  //for (auto enumID : enumUIdentifiers) {
	 // functionCodeHost << "template<parameters::" << enumID.first << " ident> auto& uGet(parameters::" << enumID.first << " = ident){\n";
	 // for (auto e : enumID.second)
		//  functionCodeHost << "\tif constexpr(ident == parameters::" << enumID.first << "::" << e.first << ") return ParameterManager::instance().uGet<" << e.second << ">(\"" << enumID.first << "." << e.first << "\");\n";
	 // functionCodeHost << "};\n";
  //}
  functionCodeCU << "\ttemplate<typename T, T ident> struct getType{using type = int32_t;};\n";
  for (auto enumID : enumIdentifiers) {
	  functionCodeCU << "\tstd::pair<std::string, std::string> getIdentifier(parameters::" << enumID.first << " ident);\n";
	  functionCodeCU << "\ttemplate<parameters::" << enumID.first << " ident> typename getType<parameters::" << enumID.first << ", ident>::type& get();\n";
	  for (auto e : enumID.second) {
		  functionCodeCU << "\ttemplate<> struct getType<parameters::" << enumID.first << ", parameters::" << enumID.first << "::" << e.first << ">{using type = " << e.second << ";};\n";
		  functionCodeCU << "\ttemplate<> typename getType<parameters::" << enumID.first << ",parameters::"<< enumID.first << "::" << e.first <<">::type& get<parameters::" << enumID.first << "::" << e.first << ">();\n";
		  functionCodeCU2 << "\ttemplate<> typename getType<parameters::" << enumID.first << ",parameters::" << enumID.first << "::" << e.first << ">::type& get<parameters::" << enumID.first << "::" << e.first << ">(){" <<
			  "\n\t\treturn ParameterManager::instance().get<" << e.second << ">(\"" << enumID.first << "." << e.first << "\");" <<
			  "\n\t}\n";
	  }
	  functionCodeCU2 << "\tstd::pair<std::string, std::string> getIdentifier(parameters::" << enumID.first << " ident){\n";
	  for (auto e : enumID.second) {
		  functionCodeCU2 << "\t\tif(ident == " << "parameters::" << enumID.first << "::" << e.first << ") return std::make_pair(std::string(\"" << enumID.first << "\"), std::string(\"" << e.first << "\"));\n";
	  }
	  functionCodeCU2 << "\t}\n";
  }
  functionCodeCU << "\ttemplate<typename T, T ident> struct getUType{using type = int32_t;};\n";
  for (auto enumID : enumUIdentifiers) {
	  functionCodeCU << "\ttemplate<parameters::" << enumID.first << " ident> typename getUType<parameters::" << enumID.first << ", ident>::type& uGet();\n";
	  for (auto e : enumID.second) {
		  functionCodeCU << "\ttemplate<> struct getUType<parameters::" << enumID.first << ", parameters::" << enumID.first << "::" << e.first << ">{using type = " << e.second << ";};\n";
		  functionCodeCU << "\ttemplate<> typename getUType<parameters::" << enumID.first << ",parameters::" << enumID.first << "::" << e.first << ">::type& uGet<parameters::" << enumID.first << "::" << e.first << ">();\n";
		  functionCodeCU2 << "\ttemplate<> typename getUType<parameters::" << enumID.first << ",parameters::" << enumID.first << "::" << e.first << ">::type& uGet<parameters::" << enumID.first << "::" << e.first << ">(){" <<
			  "\n\t\treturn ParameterManager::instance().uGet<" << e.second << ">(\"" << enumID.first << "." << e.first << "\");" <<
			  "\n\t}\n";
	  }
  }

  for (auto complexNode : complexTypes) {
	  structCode << "struct " << complexNode.get<std::string>("name") << "{" << std::endl;
	  for (auto member : complexNode.get_child("description")) {
		  auto ty = member.second.get<std::string>("type");
		  auto id = member.second.get<std::string>("identifier");
		  auto def = member.second.get<std::string>("default");
		  structCode << "\t"
			  << ty << " "
			  << id << " = "
			  << (def == "" ? "\"\"" : def) << ";" << std::endl;
	  }
	  structCode << "};" << std::endl;
  }
  for (auto complexNode : complexTypes) {
	  parameterCode2 << "ParameterManager::instance().addEncoder(typeid(" << complexNode.get<std::string>("name") << R"(), [](const detail::iAny& any) {
	const auto& var = boost::any_cast<const )" << complexNode.get<std::string>("name") << R"(&>(any);
	auto node  = YAML::Node();)" << std::endl;
	  for (auto member : complexNode.get_child("description")) {
		  auto ty = member.second.get<std::string>("type");
		  auto id = member.second.get<std::string>("identifier");
		  auto def = member.second.get<std::string>("default");
		  parameterCode2 << "\tnode[\"" << id << "\"] = callEncoder(" << ty << ", var." << id << ");\n";
	  }
	  parameterCode2 << R"(	return node;
});)" << std::endl;
  }
  for (auto complexNode : complexTypes) {
	  parameterCode2 << "ParameterManager::instance().addDecoder(typeid(" << complexNode.get<std::string>("name") << R"(), [](const YAML::Node& node) {
	 )" << complexNode.get<std::string>("name") << R"( var;)" << std::endl;
	  for (auto member : complexNode.get_child("description")) {
		  auto ty = member.second.get<std::string>("type");
		  auto id = member.second.get<std::string>("identifier");
		  auto def = member.second.get<std::string>("default");
		  parameterCode2 << "\tvar." << id << " = callDecoder(" << ty << ", node[\"" << id << "\"], var." << id << ");\n";
	  }
	  parameterCode2 << R"(	return detail::iAny(var);
});)" << std::endl;
  }
  for (auto complexNode : complexTypes) {
	  parameterCode2 << "ParameterManager::instance().addUifunction(typeid(" << complexNode.get<std::string>("name") << R"(), [](Parameter& parameter) {
	)" << complexNode.get<std::string>("name") << R"(& val = boost::any_cast<)" << complexNode.get<std::string>("name") << R"(&>(parameter.param.val.value());
	if (parameter.properties.hidden) return;
	ImGui::Text(parameter.identifier.c_str());
	ImGui::PushID(parameter.identifier.c_str());)" << std::endl;
	  for (auto member : complexNode.get_child("description")) {
		  auto ty = member.second.get<std::string>("type");
		  auto id = member.second.get<std::string>("identifier");
		  auto def = member.second.get<std::string>("default");
		  parameterCode2 << "\tcallUI(" << ty << ", val." << id << ", \"." << id << "\");\n";
	  }
	  parameterCode2 << R"(	ImGui::PopID();
});)" << std::endl;
	  parameterCode2 << "\tcustomVector(" << complexNode.get<std::string>("name") << ");" << std::endl;
  }

  std::ofstream generatedParameters(sourceDirectory + "/tools/GeneratedParameters.cpp");
  generatedParameters << R"(#include <math/math.h>
#include <tools/ParameterManager.h>
#include <utility/identifier/uniform.h>
#include <utility/identifier/arrays.h>
#include <tools/pathfinder.h>
#pragma warning(push)
#pragma warning(disable : 4251; disable : 4275)
#include <yaml-cpp/yaml.h>
#pragma warning(pop)
#include <imgui/imgui.h>

void ParameterManager::init() {
)" << parameterCode.str() << parameterCode2.str() << R"(
})" << functionCodeCU2.str();
  generatedParameters.close();

  std::ofstream generatedEnums(sourceDirectory + "/utility/identifier/uniform.h");
  generatedEnums << R"(#pragma once
#include <array>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <math/template/nonesuch.h>
#include <math/unit_math.h>
#include <utility>
#include <vector>

#include <utility/identifier/resource_helper.h>
#include <tools/ParameterManager.h>

)" << structCode.str() << "\n" << functionCode.str() << "}\n" << "\n\n" << functionCodeCU.str() << "\nnamespace parameters{ \n" << functionCodeHost.str();
  generatedEnums.close();

}
catch (std::exception e) {
	std::cerr << "Caught exception: " << e.what() << std::endl;
	throw;
}