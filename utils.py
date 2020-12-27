import pymesh

def align_tooth(toothmesh, Rs, Ts, S_pca):
    #nT = list(Ts.keys())
    #Ts.f.arr_0


    pca_mean = Ts['arr_0']
    T_pca = Ts['arr_1']
    T = Ts['arr_2']

    R_pca = Rs['arr_0']
    R = Rs['arr_1']

    return pymesh.form_mesh(((toothmesh.vertices - pca_mean) @ R_pca * S_pca + pca_mean + T_pca) @ R + T, toothmesh.faces)