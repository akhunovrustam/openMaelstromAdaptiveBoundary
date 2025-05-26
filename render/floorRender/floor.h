#pragma once
#include <render/util/renderer.h>

/** This renderer displays a simple AABB around the simulations initial domain
 * (based on the domain object and not on the current simulaltion size). If
 * parts of the simulation are open the AABB will still be drawn fully. **/
class FloorRenderer : public Renderer {
public:
	FloorRenderer();
  virtual void update() override;
  virtual void render(bool pretty) override;
  virtual bool valid() override;

  GLuint m_posAttr;
  GLuint m_colAttr;
  GLuint IBO;
  GLuint VXO;
  GLuint VUV;
};
