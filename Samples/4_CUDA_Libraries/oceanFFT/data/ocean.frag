// GLSL fragment shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;

uniform vec4 deepColor;
uniform vec4 shallowColor;
uniform vec4 skyColor;
uniform vec3 lightDir;

void main()
{
    vec3 eyeVector              = normalize(eyeSpacePos);
    vec3 eyeSpaceNormalVector   = normalize(eyeSpaceNormal);
    vec3 worldSpaceNormalVector = normalize(worldSpaceNormal);

    float facing    = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));
    float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    float diffuse   = max(0.0, dot(worldSpaceNormalVector, lightDir));
    
//    vec4 waterColor = mix(shallowColor, deepColor, facing);
    vec4 waterColor = deepColor;
    
//    gl_FragColor = gl_Color;
//    gl_FragColor = vec4(fresnel);
//    gl_FragColor = vec4(diffuse);
//    gl_FragColor = waterColor;
//    gl_FragColor = waterColor*diffuse;
    gl_FragColor = waterColor*diffuse + skyColor*fresnel;
}