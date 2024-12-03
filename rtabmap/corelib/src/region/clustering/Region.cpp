#include "rtabmap/core/region/clustering/Region.h"
#include "rtabmap/core/Signature.h"
#include <pcl/search/kdtree.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>

namespace rtabmap
{
    Region::Region(int id,
                   size_t cardinality,
                   const pcl::PointXYZ &centroid,
                   float mesh,
                   float equivalentRadius,
                   float scattering2) : _id(id),
                                        _cardinality(cardinality),
                                        _centroid(centroid),
                                        _mesh(mesh),
                                        _equivalentRadius(equivalentRadius),
                                        _scattering2(scattering2)
    {
    }

    Region *Region::create(const std::list<Signature *> &signatures,
                           float defaultScattering)
    {
        UTimer timer;
        timer.start();

        pcl::PointXYZ centroid;
        size_t cardinality = signatures.size();
        float mesh = 0;
        float equivalentRadius = 0;
        float scattering2 = 0;
        int regionId = signatures.front()->regionId();

        ULOGGER_DEBUG("Cardinality for region %d=%d", regionId, (int)cardinality);

        pcl::PointCloud<pcl::PointXYZ>::Ptr positions(new pcl::PointCloud<pcl::PointXYZ>);
        positions->resize(cardinality);

        size_t counter = 0;
        for (auto const &s : signatures)
        {
            UASSERT_MSG(s->regionId() == regionId, "Region ID mismatch in Region::create");
            pcl::PointXYZ position = s->getPose().position();

            (*positions)[counter++] = position;
            centroid.x += position.x;
            centroid.y += position.y;
            centroid.z += position.z;

        }

        centroid.x /= cardinality;
        centroid.y /= cardinality;
        centroid.z /= cardinality;       

        if (cardinality > 1)
        {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>(true));
            tree->setInputCloud(positions);

            for (const auto &p : positions->points)
            {
                std::vector<int> k_indices;
                std::vector<float> k_sqr_distances;
                int n_neighbors = tree->nearestKSearch(p, 2, k_indices, k_sqr_distances); // the 1st-NN should be the node itself, the 2nd-NN should be the nearest neighbor
                if (n_neighbors > 1)
                {
                    float nn1 = sqrt(k_sqr_distances[1]);

                    ULOGGER_DEBUG("1-NN idx=%d, dist=%f", k_indices[0], sqrt(k_sqr_distances[0]));
                    ULOGGER_DEBUG("2-NN idx=%d, dist=%f", k_indices[1], nn1);
                    ULOGGER_DEBUG("k_sqr_distances for region %d=%f", regionId, nn1);
                    mesh += nn1;
                }
                else
                {
                    // ERROR
                    UFATAL("No NN found in region");
                }
            }
            
            mesh /= cardinality;
            ULOGGER_DEBUG("Mesh for region %d=%f", regionId, mesh);

            equivalentRadius = Region::K * mesh * sqrt(cardinality);
            ULOGGER_DEBUG("Equivalent radius for region %d=%f", regionId, equivalentRadius);
            float position_distances2 = 0;
            for (const auto &p : positions->points)
            {
                position_distances2 += (pow(centroid.x - p.x, 2) +
                                        pow(centroid.y - p.y, 2) +
                                        pow(centroid.z - p.z, 2));
            }
            ULOGGER_DEBUG("Position distances 2 for region %d=%f", regionId, position_distances2);
            scattering2 = position_distances2 / (equivalentRadius + 1e-7);
        }
        else if (cardinality == 1)
        {
            scattering2 = defaultScattering;
        }

        ULOGGER_DEBUG("Time to create region %d=%fs", regionId, timer.ticks());

        return new Region(regionId,
                          cardinality,
                          centroid,
                          mesh,
                          equivalentRadius,
                          scattering2);
    }
}