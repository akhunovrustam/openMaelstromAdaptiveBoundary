#pragma once
#include <render/util/renderer.h>

/** This renderer is used to display volume boundary objects (loaded as
 * VDBs). **/
class rigidRender : public Renderer {
public:
	rigidRender(int32_t index);
  virtual void update() override;
  virtual void render(bool pretty) override;
  virtual bool valid() override;
  virtual void toggle() override;

  GLuint m_posAttr;
  GLuint m_colAttr;
  GLuint IBO;
  GLuint VXO;
  GLuint VUV;

  int32_t idx;

  Matrix4x4 modelMat;
  GLuint modelUniform = 0;

  int32_t tris = 0;

  bool active = false;
};
