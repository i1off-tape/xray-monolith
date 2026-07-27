#include "stdafx.h"
#include "ISpatial.h"

#include "xrCDB_ray_defs.h"

extern Fvector c_spatial_offset[8];

template <bool b_use_sse, bool b_first, bool b_nearest>
class _MM_ALIGN16 walker
{
public:
	ray_t ray;
	u32 mask;
	float range;
	float range2;
	ISpatial_DB* space;
public:
	walker(ISpatial_DB* _space, u32 _mask, const Fvector& _start, const Fvector& _dir, float _range)
	{
		mask = _mask;
		ray.pos.set(_start);
		ray.inv_dir.set(1.f, 1.f, 1.f).div(_dir);
		ray.fwd_dir.set(_dir);
		if (!b_use_sse)
		{
			// for FPU - zero out inf
			if (_abs(_dir.x) > flt_eps)
			{
			}
			else ray.inv_dir.x = 0;
			if (_abs(_dir.y) > flt_eps)
			{
			}
			else ray.inv_dir.y = 0;
			if (_abs(_dir.z) > flt_eps)
			{
			}
			else ray.inv_dir.z = 0;
		}
		range = _range;
		range2 = _range * _range;
		space = _space;
	}

	// fpu
	ICF BOOL _box_fpu(const Fvector& n_C, const float n_R, Fvector& coord)
	{
		// box
		float n_vR = 2 * n_R;
		Fbox BB;
		BB.set(n_C.x - n_vR, n_C.y - n_vR, n_C.z - n_vR, n_C.x + n_vR, n_C.y + n_vR, n_C.z + n_vR);
		return isect_fpu(BB.min, BB.max, ray, coord);
	}

	// sse
	ICF BOOL _box_sse(const Fvector& n_C, const float n_R, float& dist)
	{
		aabb_t box;
		/*
			float		n_vR	=		2*n_R;
			box.min.set	(n_C.x-n_vR, n_C.y-n_vR, n_C.z-n_vR);	box.min.pad = 0;
			box.max.set	(n_C.x+n_vR, n_C.y+n_vR, n_C.z+n_vR);	box.max.pad = 0;
		*/
		__m128 NR = _mm_load_ss((float*)&n_R);
		__m128 NC = _mm_unpacklo_ps(_mm_load_ss((float*)&n_C.x), _mm_load_ss((float*)&n_C.y));
		NR = _mm_add_ss(NR, NR);
		NC = _mm_movelh_ps(NC, _mm_load_ss((float*)&n_C.z));
		NR = _mm_shuffle_ps(NR, NR, _MM_SHUFFLE(1, 0, 0, 0));

		_mm_store_ps((float*)&box.min, _mm_sub_ps(NC, NR));
		_mm_store_ps((float*)&box.max, _mm_add_ps(NC, NR));

		return isect_sse(box, ray, dist);
	}

	void walk(ISpatial_NODE* N, Fvector& n_C, float n_R)
	{
		// Actual ray/aabb test
		if (b_use_sse)
		{
			// use SSE
			float d;
			if (!_box_sse(n_C, n_R, d)) return;
			if (d > range) return;
		}
		else
		{
			// use FPU
			Fvector P;
			if (!_box_fpu(n_C, n_R, P)) return;
			if (P.distance_to_sqr(ray.pos) > range2) return;
		}

		// test items
		xr_vector<ISpatial*>::iterator _it = N->items.begin();
		xr_vector<ISpatial*>::iterator _end = N->items.end();
		for (; _it != _end; _it++)
		{
			ISpatial* S = *_it;
			if (mask != (S->spatial.type & mask)) continue;
			Fsphere& sS = S->spatial.sphere;
			int quantity;
			float afT[2];
			Fsphere::ERP_Result result = sS.intersect(ray.pos, ray.fwd_dir, range, quantity, afT);

			if (result == Fsphere::rpOriginInside || ((result == Fsphere::rpOriginOutside) && (afT[0] < range)))
			{
				if (b_nearest)
				{
					switch (result)
					{
					case Fsphere::rpOriginInside: range = afT[0] < range ? afT[0] : range;
						break;
					case Fsphere::rpOriginOutside: range = afT[0];
						break;
					}
					range2 = range * range;
				}
				space->q_result->push_back(S);
				if (b_first) return;
			}
		}

		// recurse
		float c_R = n_R / 2;
		for (u32 octant = 0; octant < 8; octant++)
		{
			if (0 == N->children[octant]) continue;
			Fvector c_C;
			c_C.mad(n_C, c_spatial_offset[octant], c_R);
			walk(N->children[octant], c_C, c_R);
			if (b_first && !space->q_result->empty()) return;
		}
	}
};

void ISpatial_DB::q_ray(xr_vector<ISpatial*>& R, u32 _o, u32 _mask_and, const Fvector& _start, const Fvector& _dir,
                        float _range)
{
	cs.Enter();
	q_result = &R;
	q_result->clear_not_free();
	if (CPU::ID.feature & _CPU_FEATURE_SSE)
	{
		if (_o & O_ONLYFIRST)
		{
			if (_o & O_ONLYNEAREST)
			{
				walker<true, true, true> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
			else
			{
				walker<true, true, false> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
		}
		else
		{
			if (_o & O_ONLYNEAREST)
			{
				walker<true, false, true> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
			else
			{
				walker<true, false, false> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
		}
	}
	else
	{
		if (_o & O_ONLYFIRST)
		{
			if (_o & O_ONLYNEAREST)
			{
				walker<false, true, true> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
			else
			{
				walker<false, true, false> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
		}
		else
		{
			if (_o & O_ONLYNEAREST)
			{
				walker<false, false, true> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
			else
			{
				walker<false, false, false> W(this, _mask_and, _start, _dir, _range);
				W.walk(m_root, m_center, m_bounds);
			}
		}
	}
	cs.Leave();
}
