import numpy as np

def icosahedron_vertices():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1]
    ], dtype=np.float32)
    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=np.int32)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, faces

def subdivide_mesh(verts, faces, level):
    verts = verts.tolist()
    faces = faces.tolist()
    for _ in range(level):
        new_faces = []
        midpoint_cache = {}
        def midpoint(v1, v2):
            key = tuple(sorted((v1, v2)))
            if key not in midpoint_cache:
                mid = np.array(verts[v1]) + np.array(verts[v2])
                mid /= 2.0
                mid /= np.linalg.norm(mid)
                verts.append(mid.tolist())
                midpoint_cache[key] = len(verts)-1
            return midpoint_cache[key]
        for tri in faces:
            a, b, c = tri
            ab = midpoint(a,b)
            bc = midpoint(b,c)
            ca = midpoint(c,a)
            new_faces.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])
        faces = new_faces
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    return verts, faces

for level in [1,2,3,4]:
    verts, faces = icosahedron_vertices()
    verts, faces = subdivide_mesh(verts, faces, level)
    edges = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])
    np.savez(f"nodes_level_{level}.npz", coords=verts, edges=edges)
    print(f"nodes_level_{level}.npz created: {verts.shape[0]} nodes, {faces.shape[0]} faces")
