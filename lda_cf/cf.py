# Do not run 
# for collaborative filtering
import numpy as np


def cf(R,business_id_list):
    P = np.diag(np.diag(np.dot(R, R.T)))
    Q = np.diag(np.diag(np.dot(R.T, R)))
    p = np.diag(np.power(np.diag(P),-1/2))
    q = np.diag(np.power(np.diag(Q),-1/2))
    #collaboratvie_filtering_user_user
    collaborative_filtering_user = np.dot(np.dot(np.dot(np.dot(p,R),R.T),p),R)
    S = collaborative_filtering_user[499,0:100]
    alex = [(i,S[i]) for i in range(100)]

    sorted_alex = sorted(alex,key=lambda x: x[1],reverse = True)[0:5]
    top_movie_index = [ i[0] for i in sorted_alex]
    top_movie_score = [ i[1] for i in sorted_alex]

    print("Collaborative_filtering_user_user")
    print("The top 5 movie is:   ",[ business_id_list[a] for a in top_movie_index])
    print("Similarity is:   ",top_movie_score)

    #collaboratvie_filtering_item_item
    collaborative_filtering_user = np.dot(np.dot((np.dot((np.dot(R,q)),R.T)),R),q)
    S = collaborative_filtering_user[499,0:100]
    alex = [(i,S[i]) for i in range(100)]

    sorted_alex = sorted(alex,key=lambda x: x[1],reverse = True)[0:5]
    top_movie_index = [ i[0] for i in sorted_alex]
    top_movie_score = [ i[1] for i in sorted_alex]

    print("Collaborative_filtering_item_item")
    print("The top 5 movie is:   ",[ business_id_list[a] for a in top_movie_index])
    print("Similarity is:   ",top_movie_score)
