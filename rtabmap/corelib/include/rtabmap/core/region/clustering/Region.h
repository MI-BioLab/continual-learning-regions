#pragma once

#include "rtabmap/core/Signature.h"
#include <pcl/search/kdtree.h>
#include <pcl/common/eigen.h>
#include <pcl/common/common.h>
#include <pcl/common/point_tests.h>
#include <cmath>

namespace rtabmap
{

    class Signature;

    class Region
    {

    public:
        Region(int id,
               size_t cardinality,
               const pcl::PointXYZ &centroid,
               float mesh,
               float equivalentRadius,
               float scattering2);

        inline int id() const { return this->_id; }
        inline size_t cardinality() const { return this->_cardinality; }
        inline const pcl::PointXYZ centroid() const { return this->_centroid; }
        inline float mesh() const { return this->_mesh; }
        inline float equivalentRadius() const { return this->_equivalentRadius; }
        inline float scattering2() const { return this->_scattering2; }
        
        inline bool operator==(Region &region)
        {
            return this->_id == region.id();
        }

        static Region *create(const std::list<Signature *> &signatures, float defaultScattering);
        
        static constexpr float K = 0.525;
        static constexpr float K_2_PI = K * 2 * M_PI;

    private:
        int _id;
        size_t _cardinality;
        pcl::PointXYZ _centroid;
        float _mesh;
        float _equivalentRadius;
        float _scattering2;
    };
}

// #pragma once

// #include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines
// #include "rtabmap/core/Signature.h"
// #include <pcl/search/kdtree.h>
// #include <pcl/common/eigen.h>
// #include <pcl/common/common.h>
// #include <pcl/common/point_tests.h>
// #include <cmath>

// namespace rtabmap
// {

//     class Signature;

//     class RTABMAP_CORE_EXPORT Region
//     {

//     public:
//         Region(int id);
//         Region(int id, Signature *s);
//         Region(int id, const std::unordered_map<int, Signature *> &signatures, bool append = false);

//         ~Region();

//         inline int id() const { return this->_id; }
//         inline size_t cardinality() const { return this->_cardinality; }

//         inline const pcl::PointXYZ centroid() const { return this->_centroid; }
//         inline float mesh() const { return this->_mesh; }
//         inline float equivalentRadius() const { return this->_equivalentRadius; }
//         inline float scattering2() const { return this->_scattering2; }

//         inline const std::unordered_map<int, Signature *> &signatures() const { return this->_signatures; }
//         inline void addSignature(Signature *signature) { this->_signatures.insert({signature->id(), signature}); }
//         inline void removeSignature(int id) { this->_signatures.erase(id); }

//         inline bool operator==(Region &region)
//         {
//             return this->_id == region.id();
//         }

//         // void update(Signature *signature, float defaultScattering);

//         static Region *create(const std::unordered_map<int, Signature*> &signatures, float defaultScattering);

//         static constexpr float K = 0.525;
//         static constexpr float K_2_PI = K * 2 * M_PI;

//     private:
//         std::unordered_map<int, Signature *> _signatures;
//         int _id;
//         size_t _cardinality;
//         pcl::PointXYZ _centroid;
//         pcl::PointCloud<pcl::PointXYZ>::Ptr _positions;
//         float _mesh;
//         float _equivalentRadius;
//         float _scattering2;
//     };
// }