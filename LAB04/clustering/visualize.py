import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def run_visualization_and_save(df, X_tf, feature_cols):
    """
    Blue Lock K-Means Clustering Pipeline

    Output:
    - outputs/01_elbow.png
    - outputs/02_clustering.png
    - outputs/clustering_results.csv
    - outputs/cluster_summary.csv
    """

    print("🔄 เริ่มทำ K-Means Clustering...")


    # ===============================
    # เตรียมข้อมูล
    # ===============================

    X = X_tf.numpy()

    os.makedirs("outputs", exist_ok=True)



    # ===============================
    # 1. Elbow Method
    # ===============================

    print("📊 กำลังสร้าง Elbow Graph...")


    inertia = []

    k_range = range(1, 11)


    for k in k_range:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)

        inertia.append(kmeans.inertia_)



    plt.figure(figsize=(8,5))

    plt.plot(
        list(k_range),
        inertia,
        marker="o"
    )


    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method - Blue Lock Dataset")


    plt.savefig(
        "outputs/01_elbow.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print("✅ สร้าง 01_elbow.png แล้ว")



    # ===============================
    # 2. K-Means Model
    # ===============================

    print("⚽ กำลังจัดกลุ่มนักเตะ...")


    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )


    clusters = kmeans.fit_predict(X)



    # เพิ่ม Cluster เข้า DataFrame

    df["Cluster"] = clusters



    # ===============================
    # 3. Save Clustering Result
    # ===============================

    df.to_csv(
        "outputs/clustering_results.csv",
        index=False
    )


    print("✅ สร้าง clustering_results.csv แล้ว")



    # ===============================
    # 4. Cluster Summary
    # ===============================

    print("📋 กำลังสร้าง Cluster Summary...")


    cluster_summary = (
        df.groupby("Cluster")[feature_cols]
        .mean()
    )


    cluster_summary["Count"] = (
        df.groupby("Cluster")
        .size()
    )


    cluster_summary = (
        cluster_summary
        .reset_index()
    )



    cluster_summary.to_csv(
        "outputs/cluster_summary.csv",
        index=False
    )


    print("✅ สร้าง cluster_summary.csv แล้ว")



    # ===============================
    # 5. Cluster Visualization
    # ===============================

    print("📈 กำลังสร้างกราฟ Clustering...")


    plt.figure(figsize=(8,6))


    plt.scatter(
        X[:,0],
        X[:,1],
        c=clusters,
        cmap="viridis",
        s=80
    )


    plt.scatter(
        kmeans.cluster_centers_[:,0],
        kmeans.cluster_centers_[:,1],
        marker="X",
        s=250,
        label="Centroids"
    )


    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])

    plt.title(
        "Blue Lock Player Clustering"
    )


    plt.legend()


    plt.savefig(
        "outputs/02_clustering.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print("✅ สร้าง 02_clustering.png แล้ว")


    print("\n🎉 Blue Lock Clustering Pipeline สำเร็จ!")