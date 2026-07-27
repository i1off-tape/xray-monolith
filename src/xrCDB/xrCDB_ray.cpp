#include "stdafx.h"
#pragma hdrstop

#include "xrCDB.h"

using namespace CDB;
using namespace Opcode;

#include "xrCDB_ray_defs.h"

template <bool bUseSSE, bool bCull, bool bFirst, bool bNearest>
class _MM_ALIGN16 ray_collider
{
public:
	COLLIDER* dest;
	TRI* tris;
	Fvector* verts;

	ray_t ray;
	float rRange;
	float rRange2;

	IC void _init(COLLIDER* CL, Fvector* V, TRI* T, const Fvector& C, const Fvector& D, float R)
	{
		dest = CL;
		tris = T;
		verts = V;
		ray.pos.set(C);
		ray.inv_dir.set(1.f, 1.f, 1.f).div(D);
		ray.fwd_dir.set(D);
		rRange = R;
		rRange2 = R * R;
		if (!bUseSSE)
		{
			// for FPU - zero out inf
			if (_abs(D.x) > flt_eps)
			{
			}
			else ray.inv_dir.x = 0;
			if (_abs(D.y) > flt_eps)
			{
			}
			else ray.inv_dir.y = 0;
			if (_abs(D.z) > flt_eps)
			{
			}
			else ray.inv_dir.z = 0;
		}
	}

	// fpu
	ICF BOOL _box_fpu(const Fvector& bCenter, const Fvector& bExtents, Fvector& coord)
	{
		Fbox BB;
		BB.min.sub(bCenter, bExtents);
		BB.max.add(bCenter, bExtents);
		return isect_fpu(BB.min, BB.max, ray, coord);
	}

	// sse
	ICF BOOL _box_sse(const Fvector& bCenter, const Fvector& bExtents, float& dist)
	{
		aabb_t box;
		/*
			box.min.sub (bCenter,bExtents);	box.min.pad = 0;
			box.max.add	(bCenter,bExtents); box.max.pad = 0;
		*/
		__m128 CN = _mm_unpacklo_ps(_mm_load_ss((float*)&bCenter.x), _mm_load_ss((float*)&bCenter.y));
		CN = _mm_movelh_ps(CN, _mm_load_ss((float*)&bCenter.z));
		__m128 EX = _mm_unpacklo_ps(_mm_load_ss((float*)&bExtents.x), _mm_load_ss((float*)&bExtents.y));
		EX = _mm_movelh_ps(EX, _mm_load_ss((float*)&bExtents.z));

		_mm_store_ps((float*)&box.min, _mm_sub_ps(CN, EX));
		_mm_store_ps((float*)&box.max, _mm_add_ps(CN, EX));

		return isect_sse(box, ray, dist);
	}

	IC bool _tri(u32* p, float& u, float& v, float& range)
	{
		Fvector edge1, edge2, tvec, pvec, qvec;
		float det, inv_det;

		// find vectors for two edges sharing vert0
		Fvector& p0 = verts[p[0]];
		Fvector& p1 = verts[p[1]];
		Fvector& p2 = verts[p[2]];
		edge1.sub(p1, p0);
		edge2.sub(p2, p0);
		// begin calculating determinant - also used to calculate U parameter
		// if determinant is near zero, ray lies in plane of triangle
		pvec.crossproduct(ray.fwd_dir, edge2);
		det = edge1.dotproduct(pvec);
		if (bCull)
		{
			if (det < EPS) return false;
			tvec.sub(ray.pos, p0); // calculate distance from vert0 to ray origin
			u = tvec.dotproduct(pvec); // calculate U parameter and test bounds
			if (u < 0.f || u > det) return false;
			qvec.crossproduct(tvec, edge1); // prepare to test V parameter
			v = ray.fwd_dir.dotproduct(qvec); // calculate V parameter and test bounds
			if (v < 0.f || u + v > det) return false;
			range = edge2.dotproduct(qvec); // calculate t, scale parameters, ray intersects triangle
			inv_det = 1.0f / det;
			range *= inv_det;
			u *= inv_det;
			v *= inv_det;
		}
		else
		{
			if (det > -EPS && det < EPS) return false;
			inv_det = 1.0f / det;
			tvec.sub(ray.pos, p0); // calculate distance from vert0 to ray origin
			u = tvec.dotproduct(pvec) * inv_det; // calculate U parameter and test bounds
			if (u < 0.0f || u > 1.0f) return false;
			qvec.crossproduct(tvec, edge1); // prepare to test V parameter
			v = ray.fwd_dir.dotproduct(qvec) * inv_det; // calculate V parameter and test bounds
			if (v < 0.0f || u + v > 1.0f) return false;
			range = edge2.dotproduct(qvec) * inv_det; // calculate t, ray intersects triangle
		}
		return true;
	}

	void _prim(DWORD prim)
	{
		float u, v, r;
		if (!_tri(tris[prim].verts, u, v, r)) return;
		if (r <= 0 || r > rRange) return;

		if (bNearest)
		{
			if (dest->r_count())
			{
				RESULT& R = *dest->r_begin();
				if (r < R.range)
				{
					R.id = prim;
					R.range = r;
					R.u = u;
					R.v = v;
					R.verts[0] = verts[tris[prim].verts[0]];
					R.verts[1] = verts[tris[prim].verts[1]];
					R.verts[2] = verts[tris[prim].verts[2]];
					R.dummy = tris[prim].dummy;
					rRange = r;
					rRange2 = r * r;
				}
			}
			else
			{
				RESULT& R = dest->r_add();
				R.id = prim;
				R.range = r;
				R.u = u;
				R.v = v;
				R.verts[0] = verts[tris[prim].verts[0]];
				R.verts[1] = verts[tris[prim].verts[1]];
				R.verts[2] = verts[tris[prim].verts[2]];
				R.dummy = tris[prim].dummy;
				rRange = r;
				rRange2 = r * r;
			}
		}
		else
		{
			RESULT& R = dest->r_add();
			R.id = prim;
			R.range = r;
			R.u = u;
			R.v = v;
			R.verts[0] = verts[tris[prim].verts[0]];
			R.verts[1] = verts[tris[prim].verts[1]];
			R.verts[2] = verts[tris[prim].verts[2]];
			R.dummy = tris[prim].dummy;
		}
	}

	void _stab(const AABBNoLeafNode* node)
	{
		// Should help
		_mm_prefetch((char *)node->GetNeg(), _MM_HINT_NTA);

		// Actual ray/aabb test
		if (bUseSSE)
		{
			// use SSE
			float d;
			if (!_box_sse((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents, d)) return;
			if (d > rRange) return;
		}
		else
		{
			// use FPU
			Fvector P;
			if (!_box_fpu((Fvector&)node->mAABB.mCenter, (Fvector&)node->mAABB.mExtents, P)) return;
			if (P.distance_to_sqr(ray.pos) > rRange2) return;
		}

		// 1st chield
		if (node->HasLeaf()) _prim(node->GetPrimitive());
		else _stab(node->GetPos());

		// Early exit for "only first"
		if (bFirst && dest->r_count()) return;

		// 2nd chield
		if (node->HasLeaf2()) _prim(node->GetPrimitive2());
		else _stab(node->GetNeg());
	}
};

void COLLIDER::ray_query(const MODEL* m_def, const Fvector& r_start, const Fvector& r_dir, float r_range)
{
	m_def->syncronize();

	// Get nodes
	const AABBNoLeafTree* T = (const AABBNoLeafTree*)m_def->tree->GetTree();
	const AABBNoLeafNode* N = T->GetNodes();
	r_clear();

	if (CPU::ID.feature & _CPU_FEATURE_SSE)
	{
		// SSE
		// Binary dispatcher
		if (ray_mode & OPT_CULL)
		{
			if (ray_mode & OPT_ONLYFIRST)
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<true, true, true, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<true, true, true, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
			else
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<true, true, false, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<true, true, false, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
		}
		else
		{
			if (ray_mode & OPT_ONLYFIRST)
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<true, false, true, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<true, false, true, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
			else
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<true, false, false, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<true, false, false, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
		}
	}
	else
	{
		// FPU
		// Binary dispatcher
		if (ray_mode & OPT_CULL)
		{
			if (ray_mode & OPT_ONLYFIRST)
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<false, true, true, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<false, true, true, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
			else
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<false, true, false, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<false, true, false, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
		}
		else
		{
			if (ray_mode & OPT_ONLYFIRST)
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<false, false, true, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<false, false, true, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
			else
			{
				if (ray_mode & OPT_ONLYNEAREST)
				{
					ray_collider<false, false, false, true> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
				else
				{
					ray_collider<false, false, false, false> RC;
					RC._init(this, m_def->verts, m_def->tris, r_start, r_dir, r_range);
					RC._stab(N);
				}
			}
		}
	}
}
