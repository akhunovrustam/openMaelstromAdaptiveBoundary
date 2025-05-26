#define _CRT_SECURE_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <tools/ParameterManager.h>
#include "../imgui/imgui.h"
#include <glad/glad.h> 
#include <GLFW/glfw3.h>
#ifdef WIN32
#pragma warning(push)
#pragma warning(disable:4251; disable:4275)
#endif
#include <yaml-cpp/yaml.h>
#ifdef WIN32
#pragma warning(pop)
#endif
#include <utility/identifier/resource_helper.h>
#include <glm/glm.hpp>

namespace YAML {
#define CONVERT_V1(ty, cty)\
	template<>\
	struct convert<ty ## 1> {\
		static Node encode(const ty ## 1 & rhs) {\
			Node node;\
			node.push_back(rhs.x);\
			node.SetStyle(YAML::EmitterStyle::value::Flow);\
			return node;\
		}\
		static bool decode(const Node& node, ty ## 1& rhs) {\
			if (!node.IsSequence() || node.size() != 1) {\
				return false;\
			}\
			rhs.x = node[0].as<cty>();\
			return true;\
		}\
	};
#define CONVERT_V2(ty, cty)\
	template<>\
	struct convert<ty ## 2> {\
		static Node encode(const ty ## 2 & rhs) {\
			Node node;\
			node.push_back(rhs.x);\
			node.push_back(rhs.y);\
			node.SetStyle(YAML::EmitterStyle::value::Flow);\
			return node;\
		}\
		static bool decode(const Node& node, ty ## 2& rhs) {\
			if (!node.IsSequence() || node.size() != 2) {\
				return false;\
			}\
			rhs.x = node[0].as<cty>();\
			rhs.y = node[1].as<cty>();\
			return true;\
		}\
	};
#define CONVERT_V3(ty, cty)\
	template<>\
	struct convert<ty ## 3> {\
		static Node encode(const ty ## 3 & rhs) {\
			Node node;\
			node.push_back(rhs.x);\
			node.push_back(rhs.y);\
			node.push_back(rhs.z);\
			node.SetStyle(YAML::EmitterStyle::value::Flow);\
			return node;\
		}\
		static bool decode(const Node& node, ty ## 3& rhs) {\
			if (!node.IsSequence() || node.size() != 3) {\
				return false;\
			}\
			rhs.x = node[0].as<cty>();\
			rhs.y = node[1].as<cty>();\
			rhs.z = node[2].as<cty>();\
			return true;\
		}\
	};
#define CONVERT_V4(ty, cty)\
	template<>\
	struct convert<ty ## 4> {\
		static Node encode(const ty ## 4 & rhs) {\
			Node node;\
			node.push_back(rhs.x);\
			node.push_back(rhs.y);\
			node.push_back(rhs.z);\
			node.push_back(rhs.w);\
			node.SetStyle(YAML::EmitterStyle::value::Flow);\
			return node;\
		}\
		static bool decode(const Node& node, ty ## 4& rhs) {\
			if (!node.IsSequence() || node.size() != 4) {\
				return false;\
			}\
			rhs.x = node[0].as<cty>();\
			rhs.y = node[1].as<cty>();\
			rhs.z = node[2].as<cty>();\
			rhs.w = node[3].as<cty>();\
			return true;\
		}\
	};
	template<>
	struct convert<glm::mat4> {
		static Node encode(const glm::mat4& rhs) {
			Node node;
			node.push_back(rhs[0][0]);
			node.push_back(rhs[0][1]);
			node.push_back(rhs[0][2]);
			node.push_back(rhs[0][3]);
			node.push_back(rhs[1][0]);
			node.push_back(rhs[1][1]);
			node.push_back(rhs[1][2]);
			node.push_back(rhs[1][3]);
			node.push_back(rhs[2][0]);
			node.push_back(rhs[2][1]);
			node.push_back(rhs[2][2]);
			node.push_back(rhs[2][3]);
			node.push_back(rhs[3][0]);
			node.push_back(rhs[3][1]);
			node.push_back(rhs[3][2]);
			node.push_back(rhs[3][3]);
			node.SetStyle(YAML::EmitterStyle::value::Flow);
			return node;
		}
		static bool decode(const Node& node, glm::mat4& rhs) {
			if (!node.IsSequence() || node.size() != 16) {
				return false;
			}
			rhs[0][0] = node[0].as<float>();
			rhs[0][1] = node[1].as<float>();
			rhs[0][2] = node[2].as<float>();
			rhs[0][3] = node[3].as<float>();
			rhs[1][0] = node[0].as<float>();
			rhs[1][1] = node[1].as<float>();
			rhs[1][2] = node[2].as<float>();
			rhs[1][3] = node[3].as<float>();
			rhs[2][0] = node[0].as<float>();
			rhs[2][1] = node[1].as<float>();
			rhs[2][2] = node[2].as<float>();
			rhs[2][3] = node[3].as<float>();
			rhs[3][0] = node[0].as<float>();
			rhs[3][1] = node[1].as<float>();
			rhs[3][2] = node[2].as<float>();
			rhs[3][3] = node[3].as<float>();
			return true;
		}
	};
#define CONVERTER(ty, cty) CONVERT_V1(ty, cty) CONVERT_V2(ty, cty) CONVERT_V3(ty, cty) CONVERT_V4(ty, cty)
	CONVERTER(float, float);
	CONVERTER(double, double);
	CONVERTER(int, int32_t);
	CONVERTER(uint, uint32_t);
}

namespace detail {
	std::string to_string(float4 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + ", " + std::to_string(rhs.w) + "]";
	}
	std::string to_string(double4 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + ", " + std::to_string(rhs.w) + "]";
	}
	std::string to_string(int4 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + ", " + std::to_string(rhs.w) + "]";
	}
	std::string to_string(uint4 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + ", " + std::to_string(rhs.w) + "]";
	}

	std::string to_string(float3 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + "]";
	}
	std::string to_string(double3 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + "]";
	}
	std::string to_string(int3 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + "]";
	}
	std::string to_string(uint3 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + ", " + std::to_string(rhs.z) + "]";
	}

	std::string to_string(float2 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + "]";
	}
	std::string to_string(double2 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + "]";
	}
	std::string to_string(int2 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + "]";
	}
	std::string to_string(uint2 rhs) {
		return "[ " + std::to_string(rhs.x) + ", " + std::to_string(rhs.y) + "]";
	}
}


std::pair<std::string, std::string> split(std::string s) {
	if (s.find(".") == std::string::npos)
		return std::make_pair(std::string(""), s);
	return std::make_pair(s.substr(0, s.find(".")), s.substr(s.find(".") + 1));
}

	bool ParameterManager::parameterExists(std::string s) {
		if (parameterList.find(s) == parameterList.end()) return false;
		return true;
	}
	bool ParameterManager::isAmbiguous(std::string s) {
		auto idc = qualifiedIdentifierList.count(s);
		if (idc == 0)
			return false;
			//throw std::invalid_argument("Parameter " + id + " does not exist");
		if (idc > 1)
			return true;
			//throw std::invalid_argument("Parameter " + id + " is ambiguous");
		return false;
		//auto range = qualifiedIdentifierList.equal_range(id);
		//auto qIdentifier = range.first->second;
		//if (idc > 1)
		//	for (auto i = range.first; i != range.second; ++i)
		//		if (ns == i->second.substr(0, i->second.find(".")))
		//			qIdentifier = i->second;
		//return qIdentifier;
	}
	std::string ParameterManager::resolveParameter(std::string s) {
		if (s.find(".") != std::string::npos) return s;
		auto [ns, id] = split(s);
		auto idc = qualifiedIdentifierList.count(id);
		if (idc == 0)
			throw std::invalid_argument("Parameter " + id + " does not exist");
		if (idc > 1 && ns == "")
			throw std::invalid_argument("Parameter " + id + " is ambiguous");
		auto range = qualifiedIdentifierList.equal_range(id);
		auto qIdentifier = range.first->second;
		if (idc > 1)
			for (auto i = range.first; i != range.second; ++i)
				if (ns == i->second.substr(0, i->second.find(".")))
					qIdentifier = i->second;
		return qIdentifier;
	}
	void ParameterManager::parseTree(YAML::Node root) {
		for (auto& p : parameterList) {
				auto id = p.second->identifier;
				auto ns = p.second->identifierNamespace;
				try {
				YAML::Node node;
				bool found = false;
				if (ns != "") {
					for (auto nnode : root) {
						//std::cout << nnode.first << " : " << nnode.first.as<std::string>() << std::endl;
						//std::cout << nnode.first.IsSequence() << std::endl;
						//std::cout << nnode.first.IsMap() << std::endl;
						if (nnode.first.as<std::string>() == ns)
							if (nnode.second[id]) {
								node = nnode.second[id];
								found = true;
							}
					}
					//if (root[ns]) {
					//	auto nns = root[ns];
					//	if (nns[id])
					//		node = nns[id];
					//}
				}
				else {
					node = root[id];
					found = true;
				}
				if (found && node) {
					if (!p.second->param.isVec) {
						if (p.second->param.valVec)
						p.second->param.val.value() = decoders[p.second->type](node);
						else
						p.second->param.val = decoders[p.second->type](node);
					}
					else {
						if (!node.IsSequence() && !node.IsNull())
							throw std::invalid_argument("Expected sequence for " + p.first);
						if(p.second->param.valVec)
						p.second->param.valVec.value() = decoders[p.second->type](node);
						else
							p.second->param.valVec = decoders[p.second->type](node);
					}
				}
			}
			catch (...) {
				std::cout << ns << " : " << id << std::endl;
				throw;
			}
		}
	}

#define customVectorInternal(ty)\
	addUifunction(typeid(std::vector<ty>), [](Parameter& parameter) {\
		std::vector<ty>& vec = boost::any_cast<std::vector<ty>&>(parameter.param.valVec.value());\
		if (parameter.properties.hidden) return;\
		ImGui::PushID(parameter.identifier.c_str());\
		ImGui::Text(parameter.identifier.c_str());\
		if (!parameter.properties.constant) {\
			ImGui::SameLine();\
			if (ImGui::Button("+"))\
				vec.push_back(ty());\
			ImGui::SameLine();\
			if (ImGui::Button("-"))\
				vec.pop_back();\
		}\
		int32_t i = 0;\
		for (auto& elem : vec) {\
			ImGui::Indent();\
			ImGui::PushID(i);\
			auto eParam = Parameter{ parameter.identifier + "[" + std::to_string(i++) + "]", parameter.identifierNamespace, elem, typeid(ty), parameter.properties };\
			ParameterManager::instance().uiFunctions[typeid(ty)](eParam);\
			ImGui::Unindent();\
			ImGui::PopID();\
			if (!parameter.properties.constant)\
				elem = (ty) eParam.param.val.value();\
		}\
		ImGui::PopID();\
		});


	ParameterManager::ParameterManager() {
#define SIMPLE_PARSER(ty) \
		decoders[typeid(ty)] = [](const YAML::Node& node) {return detail::iAny(node.as<ty>()); };\
		encoders[typeid(ty)] = [](const detail::iAny& any) {return YAML::convert<ty>::encode(boost::any_cast<ty>(any)); };
#define VECTOR_PARSER(ty)\
		decoders[typeid(std::vector<ty>)] = [](const YAML::Node& node) {\
			std::vector<ty> vec;\
			for (auto e : node)\
				vec.push_back(e.as<ty>());\
			return detail::iAny(vec); \
		};\
		encoders[typeid(std::vector<ty>)] = [](const detail::iAny& any){return YAML::convert<std::vector<ty>>::encode(boost::any_cast<std::vector<ty>>(any));};
		SIMPLE_PARSER(int32_t);
		SIMPLE_PARSER(uint32_t);
		SIMPLE_PARSER(float);
		SIMPLE_PARSER(double);
		SIMPLE_PARSER(bool);
		SIMPLE_PARSER(std::string);
		SIMPLE_PARSER(float1);
		SIMPLE_PARSER(float2);
		SIMPLE_PARSER(float3);
		SIMPLE_PARSER(float4);
		SIMPLE_PARSER(double1);
		SIMPLE_PARSER(double2);
		SIMPLE_PARSER(double3);
		SIMPLE_PARSER(double4);
		SIMPLE_PARSER(int1);
		SIMPLE_PARSER(int2);
		SIMPLE_PARSER(int3);
		SIMPLE_PARSER(int4);
		SIMPLE_PARSER(uint1);
		SIMPLE_PARSER(uint2);
		SIMPLE_PARSER(uint3);
		SIMPLE_PARSER(uint4);
		SIMPLE_PARSER(glm::mat4);
		VECTOR_PARSER(int32_t);
		VECTOR_PARSER(uint32_t);
		VECTOR_PARSER(float);
		VECTOR_PARSER(double);
		VECTOR_PARSER(std::string);
		VECTOR_PARSER(float1);
		VECTOR_PARSER(float2);
		VECTOR_PARSER(float3);
		VECTOR_PARSER(float4);
		VECTOR_PARSER(double1);
		VECTOR_PARSER(double2);
		VECTOR_PARSER(double3);
		VECTOR_PARSER(double4);
		VECTOR_PARSER(int1);
		VECTOR_PARSER(int2);
		VECTOR_PARSER(int3);
		VECTOR_PARSER(int4);
		VECTOR_PARSER(uint1);
		VECTOR_PARSER(uint2);
		VECTOR_PARSER(uint3);
                VECTOR_PARSER(uint4);
                addEncoder(typeid(launch_config), [](const detail::iAny &any) {
                  const auto &var = boost::any_cast<const launch_config &>(any);
                  std::string str;
                  switch (var) {
                  case launch_config::device:
                    str = "device";
                    break;
                  case launch_config::host:
                    str = "host";
                    break;
                  case launch_config::debug:
                    str = "debug";
                    break;
                  case launch_config::pure_host:
                    str = "pure_host";
                    break;
                  case launch_config::_used_for_template_specializations:
                    str = "_used_for_template_specializations";
                    break;
                  }
                  auto node = YAML::Node();
                  node = str;
                  return node;
                });
                addEncoder(typeid(hash_length), [](const detail::iAny &any) {
                  const auto &var = boost::any_cast<const hash_length &>(any);
                  std::string str;
                  switch (var) {
                  case hash_length::bit_64:
                    str = "bit_64";
                    break;
                  case hash_length::bit_32:
                    str = "bit_32";
                    break;
                  }
                  auto node = YAML::Node();
                  node = str;
                  return node;
                });
                addEncoder(typeid(cell_ordering), [](const detail::iAny &any) {
                  const auto &var = boost::any_cast<const cell_ordering &>(any);
                  std::string str;
                  switch (var) {
                  case cell_ordering::z_order:
                    str = "z_order";
                    break;
                  case cell_ordering::linear_order:
                    str = "linear_order";
                    break;
                  }
                  auto node = YAML::Node();
                  node = str;
                  return node;
                });
                addEncoder(typeid(cell_structuring), [](const detail::iAny &any) {
                  const auto &var = boost::any_cast<const cell_structuring &>(any);
                  std::string str;
                  switch (var) {
                  case cell_structuring::hashed:
                    str = "hashed";
                    break;
                  case cell_structuring::MLM:
                    str = "MLM";
                    break;
                  case cell_structuring::complete:
                    str = "complete";
                    break;
                  case cell_structuring::compactMLM:
                    str = "compactMLM";
                    break;
                  }
                  auto node = YAML::Node();
                  node = str;
                  return node;
                });
                addEncoder(typeid(neighbor_list), [](const detail::iAny &any) {
                  const auto &var = boost::any_cast<const neighbor_list &>(any);
                  std::string str;
                  switch (var) {
                  case neighbor_list::basic:
                    str = "basic";
                    break;
                  case neighbor_list::constrained:
                    str = "constrained";
                    break;
                  case neighbor_list::cell_based:
                    str = "cell_based";
                    break;
                  case neighbor_list::compactCell:
                    str = "compactCell";
                    break;
                  case neighbor_list::compactMLM:
                    str = "compactMLM";
                    break;
                  case neighbor_list::masked:
                    str = "masked";
                    break;
                  }
                  auto node = YAML::Node();
                  node = str;
                  return node;
                });
                addDecoder(typeid(launch_config), [](const YAML::Node &node) {
                  launch_config var;
                  std::string str =
                      node
                          ? boost::any_cast<std::string>(ParameterManager::instance().decoders[typeid(std::string)](node))
                          : "debug";
                  if (str == "device")
                    var = launch_config::device;
                  if (str == "host")
                    var = launch_config::host;
                  if (str == "debug")
                    var = launch_config::debug;
                  if (str == "pure_host")
                    var = launch_config::pure_host;
                  if (str == "_used_for_template_specializations")
                    var = launch_config::_used_for_template_specializations;
                  return detail::iAny(var);
                });
                addDecoder(typeid(hash_length), [](const YAML::Node &node) {
                  hash_length var;
                  std::string str =
                      node
                          ? boost::any_cast<std::string>(ParameterManager::instance().decoders[typeid(std::string)](node))
                          : "bit_64";
                  if (str == "bit_64")
                    var = hash_length::bit_64;
                  if (str == "bit_32")
                    var = hash_length::bit_32;
                  return detail::iAny(var);
                });
                addDecoder(typeid(cell_ordering), [](const YAML::Node &node) {
                  cell_ordering var;
                  std::string str =
                      node
                          ? boost::any_cast<std::string>(ParameterManager::instance().decoders[typeid(std::string)](node))
                          : "z_order";
                  if (str == "z_order")
                    var = cell_ordering::z_order;
                  if (str == "linear_order")
                    var = cell_ordering::linear_order;
                  return detail::iAny(var);
                });
                addDecoder(typeid(cell_structuring), [](const YAML::Node &node) {
                  cell_structuring var;
                  std::string str =
                      node
                          ? boost::any_cast<std::string>(ParameterManager::instance().decoders[typeid(std::string)](node))
                          : "hashed";
                  if (str == "hashed")
                    var = cell_structuring::hashed;
                  if (str == "MLM")
                    var = cell_structuring::MLM;
                  if (str == "complete")
                    var = cell_structuring::complete;
                  if (str == "compactMLM")
                    var = cell_structuring::compactMLM;
                  return detail::iAny(var);
                });
                addDecoder(typeid(neighbor_list), [](const YAML::Node &node) {
                  neighbor_list var;
                  std::string str =
                      node
                          ? boost::any_cast<std::string>(ParameterManager::instance().decoders[typeid(std::string)](node))
                          : "basic";
                  if (str == "basic")
                    var = neighbor_list::basic;
                  if (str == "constrained")
                    var = neighbor_list::constrained;
                  if (str == "cell_based")
                    var = neighbor_list::cell_based;
                  if (str == "compactCell")
                    var = neighbor_list::compactCell;
                  if (str == "compactMLM")
                    var = neighbor_list::compactMLM;
                  if (str == "masked")
                    var = neighbor_list::masked;
                  return detail::iAny(var);
                });

                addUifunction(typeid(launch_config), [](Parameter &parameter) {
                  launch_config &val = boost::any_cast<launch_config &>(parameter.param.val.value());
                  if (parameter.properties.hidden)
                    return;
                  ImGui::PushID(parameter.identifier.c_str());
                  std::string str;
                  switch (val) {
                  case launch_config::device:
                    str = "device";
                    break;
                  case launch_config::host:
                    str = "host";
                    break;
                  case launch_config::debug:
                    str = "debug";
                    break;
                  case launch_config::pure_host:
                    str = "pure_host";
                    break;
                  case launch_config::_used_for_template_specializations:
                    str = "_used_for_template_specializations";
                    break;
                  }
                  callUI(std::string, str, parameter.identifier.c_str());
                  ImGui::PopID();
                });
                addUifunction(typeid(hash_length), [](Parameter &parameter) {
                  hash_length &val = boost::any_cast<hash_length &>(parameter.param.val.value());
                  if (parameter.properties.hidden)
                    return;
                  ImGui::PushID(parameter.identifier.c_str());
                  std::string str;
                  switch (val) {
                  case hash_length::bit_64:
                    str = "bit_64";
                    break;
                  case hash_length::bit_32:
                    str = "bit_32";
                    break;
                  }
                  callUI(std::string, str, parameter.identifier.c_str());
                  ImGui::PopID();
                });
                addUifunction(typeid(cell_ordering), [](Parameter &parameter) {
                  cell_ordering &val = boost::any_cast<cell_ordering &>(parameter.param.val.value());
                  if (parameter.properties.hidden)
                    return;
                  ImGui::PushID(parameter.identifier.c_str());
                  std::string str;
                  switch (val) {
                  case cell_ordering::z_order:
                    str = "z_order";
                    break;
                  case cell_ordering::linear_order:
                    str = "linear_order";
                    break;
                  }
                  callUI(std::string, str, parameter.identifier.c_str());
                  ImGui::PopID();
                });
                addUifunction(typeid(cell_structuring), [](Parameter &parameter) {
                  cell_structuring &val = boost::any_cast<cell_structuring &>(parameter.param.val.value());
                  if (parameter.properties.hidden)
                    return;
                  ImGui::PushID(parameter.identifier.c_str());
                  std::string str;
                  switch (val) {
                  case cell_structuring::hashed:
                    str = "hashed";
                    break;
                  case cell_structuring::MLM:
                    str = "MLM";
                    break;
                  case cell_structuring::complete:
                    str = "complete";
                    break;
                  case cell_structuring::compactMLM:
                    str = "compactMLM";
                    break;
                  }
                  callUI(std::string, str, parameter.identifier.c_str());
                  ImGui::PopID();
                });
                addUifunction(typeid(neighbor_list), [](Parameter &parameter) {
                  neighbor_list &val = boost::any_cast<neighbor_list &>(parameter.param.val.value());
                  if (parameter.properties.hidden)
                    return;
                  ImGui::PushID(parameter.identifier.c_str());
                  std::string str;
                  switch (val) {
                  case neighbor_list::basic:
                    str = "basic";
                    break;
                  case neighbor_list::constrained:
                    str = "constrained";
                    break;
                  case neighbor_list::cell_based:
                    str = "cell_based";
                    break;
                  case neighbor_list::compactCell:
                    str = "compactCell";
                    break;
                  case neighbor_list::compactMLM:
                    str = "compactMLM";
                    break;
                  case neighbor_list::masked:
                    str = "masked";
                    break;
                  }
                  callUI(std::string, str, parameter.identifier.c_str());
                  ImGui::PopID();
                });

		decoders[typeid(char const*)] = [](const YAML::Node& node) {
			auto str = node.as<std::string>();
			char* pt = new char[str.size() + 1];
			memcpy(pt, str.c_str(), str.size() + 1);
			char const* cpt = pt;
			return detail::iAny(cpt); };
		encoders[typeid(char const*)] = [](const detail::iAny& node) {
			char const* cpt = boost::any_cast<char const*>(node);
			return YAML::convert < std::string>::encode(std::string(cpt));
		};
		addUifunction(typeid(bool), [](Parameter& param) {
			bool& var = boost::any_cast<bool&>(param.param.val.value());
			if (param.properties.hidden)return;
			if (param.properties.constant) {
				int32_t ib = var ? 1 : 0;
				auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
				ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
				ImGui::SliderInt(param.identifier.c_str(), &ib, 0, 1);
				ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
				if (param.properties.description != "" && ImGui::IsItemHovered())
					ImGui::SetTooltip("%s",param.properties.description.c_str());
				return;
			}
			int32_t ib = var ? 1 : 0;
			int32_t ibb = ib;
			ImGui::SliderInt(param.identifier.c_str(), &ib, 0, 1);
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s",param.properties.description.c_str());
			if (ib != ibb)
				var = ib == 0 ? false : true;
			return;
			});
		addUifunction(typeid(std::string), [](Parameter& param) {
			std::string& var = boost::any_cast<std::string&>(param.param.val.value());
			if (param.properties.hidden)return;
			if (param.properties.constant) {
				auto vcp = var;
				auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
				ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
				static char buf1[256] = ""; 
				strcpy(buf1, var.c_str());
				ImGui::InputText(param.identifier.c_str(), buf1, 64);
				ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
				if (param.properties.description != "" && ImGui::IsItemHovered())
					ImGui::SetTooltip("%s",param.properties.description.c_str());
				return;
			}
			static char buf1[256] = "";
				strcpy(buf1, var.c_str());
			ImGui::InputText(param.identifier.c_str(), buf1, 64);
			var = buf1;

			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s",param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<std::string>&>(param.properties.presets);
				ImGui::SameLine();
				if (ImGui::BeginCombo((param.identifier + " presets").c_str(), var.c_str())) {
					for (int n = 0; n < presets.size(); n++) {
						bool is_selected = (var == presets[n]);
						if (ImGui::Selectable(presets[n].c_str(), is_selected))
							var = presets[n];
						if (is_selected)
							ImGui::SetItemDefaultFocus();
					}
					ImGui::EndCombo();
				}
			}
			});
		addUifunction(typeid(char const*), [](Parameter& param) {
			char const* var = boost::any_cast<char const*>(param.param.val.value());
			if (param.properties.hidden)return;
				auto vcp = var;
				auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
				ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
				static char buf1[256] = "";
				strcpy(buf1, var);
				ImGui::InputText(param.identifier.c_str(), buf1, 64);
				ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
				if (param.properties.description != "" && ImGui::IsItemHovered())
					ImGui::SetTooltip("%s",param.properties.description.c_str());
				return;
			});
	addUifunction(typeid(int32_t), [](Parameter& param) {
		int32_t& var = boost::any_cast<int32_t&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragInt(param.identifier.c_str(), &vcp, 0, vcp, vcp);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderInt(param.identifier.c_str(), &var, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragInt(param.identifier.c_str(), &var);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<int32_t>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), std::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (var == presets[n]);
					if (ImGui::Selectable(std::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(uint32_t), [](Parameter& param) {
		uint32_t& var = boost::any_cast<uint32_t&>(param.param.val.value());
		int32_t vari = (int32_t)var;
		int32_t varii = (int32_t)var;
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = vari;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragInt(param.identifier.c_str(), &vcp, 0, vcp, vcp);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			//ImGui::Text((param.identifier + ": " + std::to_string(var)).c_str());
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderInt(param.identifier.c_str(), &vari, (int32_t)(uint32_t)param.properties.range.value().min, (int32_t)(uint32_t)param.properties.range.value().max);
		else ImGui::DragInt(param.identifier.c_str(), &vari, 1, 0, INT_MAX);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (varii != vari) var = (uint32_t)vari;
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<uint32_t>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), std::to_string(var).c_str())) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (var == presets[n]);
					if (ImGui::Selectable(std::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(std::size_t), [](Parameter& param) {
		std::size_t& var = boost::any_cast<std::size_t&>(param.param.val.value());
		int32_t vari = (int32_t)var;
		int32_t varii = (int32_t)var;
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = vari;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragInt(param.identifier.c_str(), &vcp, 0, vcp, vcp);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			//ImGui::Text((param.identifier + ": " + std::to_string(var)).c_str());
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderInt(param.identifier.c_str(), &vari, (int32_t)(std::size_t)param.properties.range.value().min, (int32_t)(std::size_t)param.properties.range.value().max);
		else ImGui::DragInt(param.identifier.c_str(), &vari, 1, 0, INT_MAX);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (varii != vari) var = (std::size_t)vari;
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<std::size_t>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), std::to_string(var).c_str())) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (var == presets[n]);
					if (ImGui::Selectable(std::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(float), [](Parameter& param) {
		float& var = boost::any_cast<float&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat(param.identifier.c_str(), &vcp, 0, vcp, vcp, "%.5g");
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat(param.identifier.c_str(), &var, param.properties.range.value().min, param.properties.range.value().max, "%.5g");
		else ImGui::DragFloat(param.identifier.c_str(), &var, var * 0.01f, -FLT_MAX, FLT_MAX, "%.5g");
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<float>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), std::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (var == presets[n]);
					if (ImGui::Selectable(std::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(double), [](Parameter& param) {
		double& var = boost::any_cast<double&>(param.param.val.value());
		float vard = (float)var;
		float vardd = vard;
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = vard;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat(param.identifier.c_str(), &vcp, 0, vcp, vcp);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat(param.identifier.c_str(), &vard, (float)(double)param.properties.range.value().min, (float)(double)param.properties.range.value().max);
		else ImGui::DragFloat(param.identifier.c_str(), &vard, vard * 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (vard != vardd)
			var = vard;
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<double>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), std::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (var == presets[n]);
					if (ImGui::Selectable(std::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});

	addUifunction(typeid(float4), [](Parameter& param) {
		float4& var = boost::any_cast<float4&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat4(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat4(param.identifier.c_str(), &var.x, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragFloat4(param.identifier.c_str(), &var.x, 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<float4>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(double4), [](Parameter& param) {
		double4& vard = boost::any_cast<double4&>(param.param.val.value());
		float4 var{ (float)vard.x, (float)vard.y, (float)vard.z,(float)vard.w };
		float4 varb = var;
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat4(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat4(param.identifier.c_str(), &var.x, (float)(double)param.properties.range.value().min, (float)(double)param.properties.range.value().max);
		else ImGui::DragFloat4(param.identifier.c_str(), &var.x, 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (varb.x != var.x || varb.y != var.y || varb.z != var.z || varb.w != var.w) vard = double4{ (double)var.x, (double)var.y,(double)var.z, (double)var.w };
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<double4>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						vard = double4{ presets[n].x, presets[n].y, presets[n].z, presets[n].w };
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});

	addUifunction(typeid(int4), [](Parameter& param) {
		int4& var = boost::any_cast<int4&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragInt4(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderInt4(param.identifier.c_str(), &var.x, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragInt4(param.identifier.c_str(), &var.x, 1.f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<int4>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(float3), [](Parameter& param) {
		float3& var = boost::any_cast<float3&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat3(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat3(param.identifier.c_str(), &var.x, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragFloat3(param.identifier.c_str(), &var.x, 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<float3>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(double3), [](Parameter& param) {
		double3& vard = boost::any_cast<double3&>(param.param.val.value());
		float3 var{ (float)vard.x, (float)vard.y, (float)vard.z };
		float3 varb = var;
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat3(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat3(param.identifier.c_str(), &var.x, (float)(double)param.properties.range.value().min, (float)(double)param.properties.range.value().max);
		else ImGui::DragFloat3(param.identifier.c_str(), &var.x, 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (varb.x != var.x || varb.y != var.y || varb.z != var.z) vard = double3{ (double)var.x, (double)var.y,(double)var.z };
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<double3>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						vard = double3{ presets[n].x, presets[n].y, presets[n].z };
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});

	addUifunction(typeid(int3), [](Parameter& param) {
		int3& var = boost::any_cast<int3&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragInt3(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderInt3(param.identifier.c_str(), &var.x, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragInt3(param.identifier.c_str(), &var.x, 1.f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<int3>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(float2), [](Parameter& param) {
		float2& var = boost::any_cast<float2&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat2(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat2(param.identifier.c_str(), &var.x, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragFloat2(param.identifier.c_str(), &var.x, 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<float2>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
	addUifunction(typeid(double2), [](Parameter& param) {
		double2& vard = boost::any_cast<double2&>(param.param.val.value());
		float2 var{ (float)vard.x, (float)vard.y };
		float2 varb = var;
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragFloat2(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderFloat2(param.identifier.c_str(), &var.x, (float)(double)param.properties.range.value().min, (float)(double)param.properties.range.value().max);
		else ImGui::DragFloat2(param.identifier.c_str(), &var.x, 0.01f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (varb.x != var.x || varb.y != var.y) vard = double2{ (double)var.x, (double)var.y };
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<double2>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						vard = double2{ presets[n].x, presets[n].y };
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});

	addUifunction(typeid(int2), [](Parameter& param) {
		int2& var = boost::any_cast<int2&>(param.param.val.value());
		if (param.properties.hidden)return;
		if (param.properties.constant) {
			auto vcp = var;
			auto col = ImGui::GetStyle().Colors[ImGuiCol_FrameBg];
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
			ImGui::DragInt2(param.identifier.c_str(), &vcp.x, 0, 0, 0);
			ImGui::GetStyle().Colors[ImGuiCol_FrameBg] = col;
			if (param.properties.description != "" && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", param.properties.description.c_str());
			return;
		}
		if (param.properties.range)
			ImGui::SliderInt2(param.identifier.c_str(), &var.x, param.properties.range.value().min, param.properties.range.value().max);
		else ImGui::DragInt2(param.identifier.c_str(), &var.x, 1.f);
		if (param.properties.description != "" && ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", param.properties.description.c_str());
		if (!param.properties.presets.empty()) {
			auto& presets = boost::any_cast<std::vector<int2>&>(param.properties.presets);
			ImGui::SameLine();
			if (ImGui::BeginCombo((param.identifier + " presets").c_str(), detail::to_string(var).c_str(), ImGuiComboFlags_HeightLarge)) {
				for (int n = 0; n < presets.size(); n++) {
					bool is_selected = (detail::to_string(var) == detail::to_string(presets[n]));
					if (ImGui::Selectable(detail::to_string(presets[n]).c_str(), is_selected))
						var = presets[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
		}
		});
		customVectorInternal(int32_t);
		customVectorInternal(float);
	/*	addUifunction(typeid(std::vector<int32_t>), [](Parameter& param) {
			std::vector<int32_t>& var = boost::any_cast<std::vector<int32_t>&>(param.param.valVec.value());
			if (param.properties.hidden)return;
			if (param.properties.constant) 
			{
				ImGui::Text(param.identifier.c_str());
				int32_t i = 0;
				for (auto e : var) {
					ImGui::DragInt(("##" + param.identifier + std::to_string(i++)).c_str(), &e);
				}
			}
			ImGui::Text(param.identifier.c_str());
			ImGui::SameLine();
			if (ImGui::Button("+"))
				var.push_back(int32_t());
			ImGui::SameLine();
			if (ImGui::Button("-"))
				var.pop_back();
			int32_t i = 0;
			for (auto& e : var) {
				ImGui::DragInt(("##" + param.identifier + std::to_string(i++)).c_str(), &e);
			}
			
			});
		addUifunction(typeid(std::vector<float>), [](Parameter& param) {
			std::vector<float>& var = boost::any_cast<std::vector<float>&>(param.param.valVec.value());
			if (param.properties.hidden)return;
			if (param.properties.constant)
			{
				ImGui::Text(param.identifier.c_str());
				int32_t i = 0;
				for (auto e : var) {
					ImGui::DragFloat(("##" + param.identifier + std::to_string(i++)).c_str(), &e,e * 0.01f);
				}
			}
			ImGui::Text(param.identifier.c_str());
			ImGui::SameLine();
			if (ImGui::Button("+"))
				var.push_back(float());
			ImGui::SameLine();
			if (ImGui::Button("-"))
				var.pop_back();
			int32_t i = 0;
			for (auto& e : var) {
				ImGui::DragFloat(("##" + param.identifier + std::to_string(i++)).c_str(), &e, fabsf(e) > 1e-5f ? fabsf(e) * 0.01f : 1e-5f, -FLT_MAX, FLT_MAX);
			}

			});*/

	}
	void ParameterManager::addDecoder(std::type_index ty, std::function<detail::iAny(const YAML::Node&)> fn) { decoders[ty] = fn; }
	void ParameterManager::addEncoder(std::type_index ty, std::function<YAML::Node(const detail::iAny&)> fn) { encoders[ty] = fn; }
	void ParameterManager::addUifunction(std::type_index ty, std::function<void(Parameter&)> fn) { uiFunctions[ty] = fn; }
	ParameterManager& ParameterManager::instance() {
		static ParameterManager inst;
		static bool once = true;
		return inst;
	}
	Parameter& ParameterManager::getParameter(std::string identifier) {
		auto qid = resolveParameter(identifier);
		if (!parameterExists(qid)) throw std::invalid_argument("Parameter " + identifier + " does not exist");
		return *parameterList[qid];
	}
	detail::varAny& ParameterManager::get(std::string identifier) {
		return getParameter(identifier).param;
	}
	void ParameterManager::load(std::string filename) {
		namespace fs = std::filesystem;
		if (!fs::exists(filename))
			throw std::invalid_argument("File does not exist");

		auto config = YAML::LoadFile(fs::path(filename).string());
		parseTree(config);
	}
	void ParameterManager::loadDirect(std::string yaml) {
		parseTree(YAML::Load(yaml));
	}
	void ParameterManager::loadTree(YAML::Node yaml) {
		parseTree(yaml);
	}
	YAML::Node ParameterManager::buildTree() {
		YAML::Node root;
		int32_t i = 0;
		for (auto p : parameterList) {
			auto id = p.second->identifier;
			auto ns = p.second->identifierNamespace;
			if (ns != "")
				root[ns][id] = encoders[p.second->type](p.second->param.isVec ? p.second->param.valVec.value() : p.second->param.val.value());
			else
				root[id] = encoders[p.second->type](p.second->param.isVec ? p.second->param.valVec.value() : p.second->param.val.value());
			//if (i++ > 50)break;
		}
		return root;
	}
	void ParameterManager::buildImguiWindow(bool* p_open) {
		{
			if (!ImGui::Begin("Parameter Manager", p_open))
			{
				ImGui::End();
				return;
			}
			std::map<std::string, std::vector<std::pair<std::string, Parameter*>>> parameters;
			for (auto p : parameterList) {
				auto [ns, id] = split(p.first);
				parameters[ns].push_back(std::make_pair(id, p.second));
			}
			for (auto param : parameters) {
				if (param.first == "" ? true : ImGui::CollapsingHeader(param.first.c_str())) {
					ImGui::PushID(param.first.c_str());
					for (auto p : param.second) {
						if (uiFunctions.find(p.second->type) == uiFunctions.end()) {
							ImGui::Text(p.first.c_str());
						}
						else {
							uiFunctions[p.second->type](*p.second);
						}
					}
					ImGui::PopID();
				}
			}
			ImGui::End();
		}
	}


void gvdbHelper::initialize() {
	return;
	static VolumeGVDB gvdbInstance;
	gvdbInstance.SetDebug(false);
	gvdbInstance.SetVerbose(false); // enable/disable console output from gvdb
	gvdbInstance.SetCudaDevice(GVDB_DEV_FIRST);
	gvdbInstance.Initialize();
}

std::vector<nvdb::VolumeGVDB> gvdbHelper::gvdbInstances;


#include <utility/identifier/uniform.h>
nvdb::VolumeGVDB& gvdbHelper::getNewInstance() { 
	auto& instance = gvdbInstances.emplace_back();
	instance.SetDebug(false);
	instance.SetVerbose(false); // enable/disable console output from gvdb
	instance.SetCudaDevice(GVDB_DEV_CURRENT);
	instance.Initialize();

	instance.AddPath(get<parameters::internal::working_directory>().c_str());
	instance.AddPath(get < parameters::internal::binary_directory>().c_str());
	instance.AddPath(get < parameters::internal::source_directory>().c_str());
	instance.AddPath(get < parameters::internal::build_directory>().c_str());
	instance.AddPath(get < parameters::internal::config_folder>().c_str());	
	return instance; 
}


template <typename Tag, typename Tag::type M> struct Rob {
	friend typename Tag::type get(Tag) { return M; }
};

// tag used to access A::member
struct A_member {
	typedef CUdeviceptr VolumeGVDB::* type;
	friend type get(A_member);
};

template struct Rob<A_member, &VolumeGVDB::cuVDBInfo>;

std::vector<VolumeGVDB> gvdbVolumeManagers;

#include <openvdb/openvdb.h>
#include <tools/pathfinder.h>

std::pair<float4, float4> getGVDBTransform(std::string fileName) {
	typedef openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<float, 4>, 3>, 3>, 3>>> FloatTree34;
	typedef openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<openvdb::Vec3f, 4>, 3>, 3>, 3>>> Vec3fTree34;
	typedef openvdb::Grid<FloatTree34>		FloatGrid34;
	typedef openvdb::Grid<Vec3fTree34>		Vec3fGrid34;
	typedef FloatGrid34						GridType34;
	typedef FloatGrid34::TreeType			TreeType34F;
	typedef Vec3fGrid34::TreeType			TreeType34VF;
	typedef openvdb::FloatGrid				FloatGrid543;
	typedef openvdb::Vec3fGrid				Vec3fGrid543;
	typedef FloatGrid543					GridType543;
	typedef FloatGrid543::TreeType			TreeType543F;
	typedef Vec3fGrid543::TreeType			TreeType543VF;

	FloatGrid543::Ptr			grid543F;
	Vec3fGrid543::Ptr			grid543VF;
	FloatGrid34::Ptr			grid34F;
	Vec3fGrid34::Ptr			grid34VF;
	// iterators
	TreeType543F::LeafCIter		iter543F;
	TreeType543VF::LeafCIter	iter543VF;
	TreeType34F::LeafCIter		iter34F;
	TreeType34VF::LeafCIter		iter34VF;


	auto vdbSkip = [&](int leaf_start, int gt, bool isFloat)
	{
		switch (gt) {
		case 0:
			if (isFloat) { iter543F = grid543F->tree().cbeginLeaf();  for (int j = 0; iter543F && j < leaf_start; j++) ++iter543F; }
			else { iter543VF = grid543VF->tree().cbeginLeaf(); for (int j = 0; iter543VF && j < leaf_start; j++) ++iter543VF; }
			break;
		case 1:
			if (isFloat) { iter34F = grid34F->tree().cbeginLeaf();  for (int j = 0; iter34F && j < leaf_start; j++) ++iter34F; }
			else { iter34VF = grid34VF->tree().cbeginLeaf(); for (int j = 0; iter34VF && j < leaf_start; j++) ++iter34VF; }
			break;
		};
	};
	auto vdbCheck = [&](int gt, bool isFloat)
	{
		switch (gt) {
		case 0: return (isFloat ? iter543F.test() : iter543VF.test());	break;
		case 1: return (isFloat ? iter34F.test() : iter34VF.test());	break;
		};
		return false;
	};
	auto vdbOrigin = [&](openvdb::Coord& orig, int gt, bool isFloat)
	{
		switch (gt) {
		case 0: if (isFloat) (iter543F)->getOrigin(orig); else (iter543VF)->getOrigin(orig);	break;
		case 1: if (isFloat) (iter34F)->getOrigin(orig); else (iter34VF)->getOrigin(orig);	break;
		};
	};
	auto vdbNext = [&](int gt, bool isFloat)
	{
		switch (gt) {
		case 0: if (isFloat) iter543F.next();	else iter543VF.next();		break;
		case 1: if (isFloat) iter34F.next();	else iter34VF.next();		break;
		};
	};


	openvdb::CoordBBox box;
	openvdb::Coord orig;
	Vector3DF p0, p1;

	PERF_PUSH("Clear grid");

	PERF_POP();

	PERF_PUSH("Load VDB");

	// Read .vdb file	

	openvdb::io::File* vdbfile = new openvdb::io::File(fileName);
	vdbfile->open();

	// Read grid		
	openvdb::GridBase::Ptr baseGrid;
	openvdb::io::File::NameIterator nameIter = vdbfile->beginName();
	std::string name = vdbfile->beginName().gridName();
	baseGrid = vdbfile->readGrid(name);
	PERF_POP();

	// Initialize GVDB config
	Vector3DF voxelsize;
	int gridtype = 0;

	bool isFloat = false;

	if (baseGrid->isType< FloatGrid543 >()) {
		gridtype = 0;
		isFloat = true;
		grid543F = openvdb::gridPtrCast< FloatGrid543 >(baseGrid);
		voxelsize.Set(grid543F->voxelSize().x(), grid543F->voxelSize().y(), grid543F->voxelSize().z());
	}
	if (baseGrid->isType< Vec3fGrid543 >()) {
		gridtype = 0;
		isFloat = false;
		grid543VF = openvdb::gridPtrCast< Vec3fGrid543 >(baseGrid);
		voxelsize.Set(grid543VF->voxelSize().x(), grid543VF->voxelSize().y(), grid543VF->voxelSize().z());
	}
	if (baseGrid->isType< FloatGrid34 >()) {
		gridtype = 1;
		isFloat = true;
		grid34F = openvdb::gridPtrCast< FloatGrid34 >(baseGrid);
		voxelsize.Set(grid34F->voxelSize().x(), grid34F->voxelSize().y(), grid34F->voxelSize().z());
	}
	if (baseGrid->isType< Vec3fGrid34 >()) {
		gridtype = 1;
		isFloat = false;
		grid34VF = openvdb::gridPtrCast< Vec3fGrid34 >(baseGrid);
		voxelsize.Set(grid34VF->voxelSize().x(), grid34VF->voxelSize().y(), grid34VF->voxelSize().z());
	}

	slong leaf;
	int leaf_start = 0;				// starting leaf		gScene.mVLeaf.x;		
	int n, leaf_max, leaf_cnt = 0;
	Vector3DF vclipmin{ -FLT_MAX, -FLT_MAX, -FLT_MAX }, vclipmax{ FLT_MAX, FLT_MAX, FLT_MAX }, voffset;
	Vector3DF mVoxMin, mVoxMax;
	// Determine Volume bounds
	vdbSkip(leaf_start, gridtype, isFloat);
	for (leaf_max = 0; vdbCheck(gridtype, isFloat); ) {
		vdbOrigin(orig, gridtype, isFloat);
		p0.Set(orig.x(), orig.y(), orig.z());
		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {		// accept condition
			if (leaf_max == 0) {
				mVoxMin = p0; mVoxMax = p0;
			}
			else {
				if (p0.x < mVoxMin.x) mVoxMin.x = p0.x;
				if (p0.y < mVoxMin.y) mVoxMin.y = p0.y;
				if (p0.z < mVoxMin.z) mVoxMin.z = p0.z;
				if (p0.x > mVoxMax.x) mVoxMax.x = p0.x;
				if (p0.y > mVoxMax.y) mVoxMax.y = p0.y;
				if (p0.z > mVoxMax.z) mVoxMax.z = p0.z;
			}
			leaf_max++;
		}
		vdbNext(gridtype, isFloat);
	}
	voffset = mVoxMin * -1;		// offset to positive space (hack)	

	return std::make_pair(float4{ voffset.x, voffset.y, voffset.z ,0.f }, 1.f / float4{ voxelsize.x, voxelsize.y, voxelsize.z,1.f });
}


std::tuple< VDBInfo*, float4, float4> gvdbHelper::loadIntoGVDB(std::string file) { 
	auto& instance = getNewInstance();

	auto fileName = resolveFile(file, { get<parameters::internal::config_folder>() });
	std::string vdbFile = fileName.string();
	char scnpath[1024];
	if (!instance.getScene()->FindFile(fileName.string().c_str(), scnpath)) {
		gprintf("Cannot find vdb file.\n");
		gerror();
	}
	//printf("Loading VDB. %s\n", scnpath);
	instance.SetChannelDefault(16, 16, 1);
	instance.getScene()->mVClipMin = nvdb::Vector3DF(-1e10f, -1e10f, -1e10f);
	instance.getScene()->mVClipMax = nvdb::Vector3DF(1e10f, 1e10f, 1e10f);
	if (!instance.LoadVDB(scnpath)) { // Load OpenVDB format
		gerror();
	}
	instance.PrepareVDB();
	instance.Measure(true);
	auto gptr = reinterpret_cast<VDBInfo*>(instance.*get(A_member()));
	Matrix4x4 itow, wtoi;
	auto [offset, size] = getGVDBTransform(scnpath);
	
	return std::make_tuple(gptr, offset, size);
}
