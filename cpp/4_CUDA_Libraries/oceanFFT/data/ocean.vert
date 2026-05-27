// GLSL vertex shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
uniform float heightScale; // = 0.5;
uniform float chopiness;   // = 1.0;
uniform vec2  size;        // = vec2(256.0, 256.0);

void main()
{
    float height     = gl_MultiTexCoord0.x;
    vec2  slope      = gl_MultiTexCoord1.xy;

    // calculate surface normal from slope for shading
	vec3 normal      = normalize(cross( vec3(0.0, slope.y*heightScale, 2.0 / size.x), vec3(2.0 / size.y, slope.x*heightScale, 0.0)));
    worldSpaceNormal = normal;

    // calculate position and transform to homogeneous clip space
    vec4 pos         = vec4(gl_Vertex.x, height * heightScale, gl_Vertex.z, 1.0);
    gl_Position      = gl_ModelViewProjectionMatrix * pos;
    
    eyeSpacePos      = (gl_ModelViewMatrix * pos).xyz;
    eyeSpaceNormal   = (gl_NormalMatrix * normal).xyz;
}
