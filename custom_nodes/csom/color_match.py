#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
import numpy as np
from skimage import color

# img_rgb: numpy array (H, W, 3), float in [0, 1] or uint8
def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.dtype == np.uint8:
        img = img_rgb.astype(np.float32) / 255.0
    else:
        img = img_rgb.astype(np.float32)

    lab = color.rgb2lab(img)   # (H, W, 3), L* in [0, 100]
    return lab

def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    rgb = color.lab2rgb(img_lab)   # float in [0,1]
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb8

def color_match(tgt, img, ref):
    # img = rgb_to_lab(img)
    # ref = rgb_to_lab(ref)
    params = build_params(img, ref)
    res = apply_params(tgt, params)
    # res = lab_to_rgb(res)
    return res

def build_params(src, ref):
    """
    This function computes the transfer matrix based on the Monge-Kantorovich Linearization (MKL).
    """
    # r = src[src.reshape()[..., 3] > 0.5, :3].T
    # z = ref[ref[..., 3] > 0.5, :3].T
    r = src.reshape((-1, 4))
    r = r[r[:, 3]>0.5]
    r = r[:, :3].T

    mu_r = r.mean(axis=-1, keepdims=True)

    z = ref.reshape((-1, 4))
    z = z[z[:, 3]>0.5]
    z = z[:, :3].T

    mu_z = z.mean(axis=-1, keepdims=True)


    print(r.shape, z.shape)
    # r = np.concatenate((r, 0.3*np.ones((3, 100* r.shape[1])), -1*np.ones((3, 100* r.shape[1]))), axis=1)
    # z = np.concatenate((z, np.ones((3, 100* z.shape[1])), -0.5*np.ones((3, 100* z.shape[1]))), axis=1)

    cov_r, cov_z = np.cov(r), np.cov(z)
    print(cov_r.shape, cov_z.shape)
    print(r.shape, z.shape)
    print("mu_r", mu_r)
    print("mu_z", mu_z)

    eig_val_r, eig_vec_r = np.linalg.eig(cov_r)
    eig_val_r[eig_val_r < 0] = 0
    val_r = np.diag(np.sqrt(eig_val_r[::-1]))
    vec_r = np.array(eig_vec_r[:, ::-1])
    inv_r = np.diag(1. / (np.diag(val_r + np.spacing(1))))

    mat_c = val_r @ vec_r.T @ cov_z @ vec_r @ val_r
    eig_val_c, eig_vec_c = np.linalg.eig(mat_c)
    eig_val_c[eig_val_c < 0] = 0
    val_c = np.diag(np.sqrt(eig_val_c))

    transfer_mat = vec_r @ inv_r @ eig_vec_c @ val_c @ eig_vec_c.T @ inv_r @ vec_r.T
    # transfer_mat = np.eye(3)
    mu_shift = mu_z - np.dot(transfer_mat, mu_r)
    print("#KES2#", type(transfer_mat), type(mu_shift))
    return transfer_mat, mu_shift


def apply_params(src: np.ndarray, transfer_params) -> np.ndarray:
    print("#KES#", src.shape)
    r = src.reshape((-1, 4))
    # r = r[r[:, 3]>0.5]
    res = r.copy().T
    r = r[:, :3].T
    mu_r = r.mean(axis=-1, keepdims=True)

    # print(r.shape)
    # print("mu_r2", mu_r)

    transfer_mat, mu_shift = transfer_params
    # # transfer the intensity distributions
    print("#KES#", type(transfer_mat), type(r), type(mu_shift), type(res))
    res[:3, :] = np.dot(transfer_mat, r) + mu_shift
    # res[:3, :] = r

    # reshape pixel array
    res = res.T.reshape(src.shape)
    print("#KES#", res.shape)

    return res



    # def __init__(self, *args, **kwargs):
    #     super(TransferMVGD, self).__init__(*args, **kwargs)

    #     # extract method from kwargs (if available)
    #     self._fun_dict = {'mvgd': self.analytical_solver, 'mkl': self.mkl_solver}
    #     try:
    #         self._fun_name = [kw for kw in list(self._fun_dict.keys()) if kwargs['method'].__contains__(kw)][0]
    #     except (BaseException, IndexError):
    #         # default function
    #         self._fun_name = 'mkl'
    #     self._fun_call = self._fun_dict[self._fun_name] if self._fun_name in self._fun_dict else self.mkl_solver

    #     # initialize variables
    #     self.r, self.z, self.cov_r, self.cov_z, self.mu_r, self.mu_z, self.transfer_mat = [None]*7

    # def init_vars(self):

    #     # reshape source and reference images
    #     self.r, self.z = self._src.reshape([-1, self._src.shape[2]]).T, self._ref.reshape([-1, self._ref.shape[2]]).T

    #     # compute covariance matrices
    #     self.cov_r, self.cov_z = np.cov(self.r), np.cov(self.z)

    #     # compute color channel means
    #     self.mu_r, self.mu_z = self.r.mean(axis=1)[..., np.newaxis], self.z.mean(axis=1)[..., np.newaxis]

    #     # validate dimensionality
    #     self.check_dims()

    # def analytical_solver(self) -> np.ndarray:
    #     """
    #     An analytical solution to the linear equation system of Multi-Variate Gaussian Distributions (MVGDs).

    #     :return: **transfer_mat**: Transfer matrix
    #     :type transfer_mat: :class:`~numpy:numpy.ndarray`
    #     :rtype: np.ndarray

    #     """

    #     # validate dimensionality
    #     self.check_dims()
    #     if self.r.shape[-1] != self.z.shape[-1]:
    #         raise Exception('Analytical MVGD solution requires spatial dimensions of both images to be equal')

    #     cov_r_inv = np.linalg.pinv(self.cov_r)
    #     cov_z_inv = np.linalg.pinv(self.cov_z)

    #     # compute transfer matrix using analytical method
    #     self.transfer_mat = np.linalg.pinv((self.z-self.mu_z).T @ cov_z_inv) @ (self.r-self.mu_r).T @ cov_r_inv

    #     return self.transfer_mat



# ------------------------------------------------------------
# Image I/O
# ------------------------------------------------------------

def load_img(path: str) -> np.ndarray:
    """
    Load image as float32 NumPy array in [0,1], shape (H,W,3).
    """
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr[:,:,:3] *= arr[:,:,3:]
    return arr


def save_img(arr: np.ndarray, path: str):
    """
    Save float32 (H,W,3) array in [0,1] to disk.
    """
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ------------------------------------------------------------
# Color transfer wrapper
# ------------------------------------------------------------

def hist_match_images(src_path: str, ref_path: str, out_path: str, method: str = "hm"):
    """
    Apply color transfer using color-matcher library.
    method:
        "hm"      - histogram matching
        "reinhard"
        "monge-kantorovich"
        "mkl"
        "mvgd"
    """

    # load images
    src = load_img(src_path)
    ref = load_img(ref_path)

    matched = color_match(src, src, ref) # ColorMatcher().transfer(src[:,:,:3], ref[:,:,:3], method=method)

    # save result
    save_img(matched, out_path)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:
        script.py SRC REF OUT [method]

    example:
        python color_transfer.py src.png ref.png out.png hm
        python color_transfer.py a.jpg b.jpg out.png reinhard
    """

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python color_transfer.py SRC REF OUT [method]")
        sys.exit(1)

    src_path = sys.argv[1]
    ref_path = sys.argv[2]
    out_path = sys.argv[3]
    method   = sys.argv[4] if len(sys.argv) == 5 else "hm"

    hist_match_images(src_path, ref_path, out_path, method)


import numpy as np

def build_lut(src, dst, n=33, round_index=True, smooth_after=True, smooth_passes=1):
    """
    Build a 3D LUT (n,n,n,3) that approximates f^{-1} by binning samples:
      - for each pair (src[i] -> dst[i]), accumulate src[i] into the cell
        corresponding to dst[i] (output space), then average per cell
      - diffuse-fill unobserved cells with fast 6-neighbor averaging

    Parameters
    ----------
    src : (m,3) array-like in [0,1]    # inputs a_i
    dst : (m,3) array-like in [0,1]    # outputs b_i = f(a_i)
    n : int                             # LUT edge resolution
    round_index : bool                  # True: round to nearest cell; False: floor
    smooth_after : bool                 # optional small smoothing after fill
    smooth_passes : int                 # number of box-smooth passes

    Returns
    -------
    lut : (n,n,n,3) float32 in [0,1]
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3

    # --- 1) Scatter-accumulate into output-space grid ---
    idxf = dst * (n - 1)
    idx = np.rint(idxf) if round_index else np.floor(idxf)
    idx = np.clip(idx, 0, n - 1).astype(np.int64)

    sums = np.zeros((n, n, n, 3), dtype=np.float64)
    cnts = np.zeros((n, n, n), dtype=np.int64)

    # vectorized scatter-add
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    np.add.at(sums, (x, y, z, slice(None)), src)
    np.add.at(cnts, (x, y, z), 1)

    # average where we have observations
    lut = np.zeros_like(sums, dtype=np.float64)
    seen = cnts > 0
    lut[seen] = sums[seen] / cnts[seen, None]

    # --- 2) Diffuse-fill holes (fast two-pass style; repeated until filled) ---
    mask = seen.copy()
    if not mask.all():
        lut = diffuse_fill_6n(lut, mask, max_iters=3*n)  # enough to reach entire grid

    # # --- 3) Optional small smoothing to reduce seams ---
    # if smooth_after and smooth_passes > 0:
    #     for _ in range(smooth_passes):
    #         lut = box_blur_6n(lut)

    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def diffuse_fill_6n(lut, mask, max_iters=99):
    """
    Fill NaN/unknown voxels by repeatedly averaging over the 6-neighborhood.
    mask: boolean array (n,n,n) True where lut is known.
    """
    lut = lut.copy()
    mask = mask.copy()

    n = lut.shape[0]
    assert lut.shape[:3] == mask.shape and lut.shape[3] == 3

    for _ in range(max_iters):
        if mask.all():
            break

        # Sum neighbor values and weights (6-connectivity)
        sum_vals = np.zeros_like(lut)
        sum_w = np.zeros(mask.shape, dtype=np.int32)

        for ax, sh in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
            v = np.roll(lut, sh, axis=ax)
            m = np.roll(mask, sh, axis=ax)
            sum_vals += v * m[..., None]
            sum_w += m.astype(np.int32)

        can_fill = (~mask) & (sum_w > 0)
        if not np.any(can_fill):
            # Nothing to pull from; fallback to global mean of known cells
            mean_val = lut[mask].reshape(-1, 3).mean(axis=0) if mask.any() else np.array([0.5, 0.5, 0.5])
            lut[~mask] = mean_val
            mask[:] = True
            break

        lut[can_fill] = (sum_vals[can_fill] / sum_w[can_fill, None])
        mask[can_fill] = True

    return lut


def box_blur_6n(lut):
    """
    One pass of normalized 6-neighbor box blur (does not shrink gamut).
    """
    n = lut.shape[0]
    out = lut.copy()
    wsum = np.ones((n, n, n), dtype=np.float64)  # center weight = 1
    acc = lut.copy()

    for ax, sh in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
        v = np.roll(lut, sh, axis=ax)
        acc += v
        wsum += 1.0

    out = acc / wsum[..., None]
    return out

def apply_lut(img, LUT):
    """
    img: (H,W,3) float in [0,1]
    LUT: (n,n,n,3)
    """
    n = LUT.shape[0]
    # scale to [0, n-1]
    pos = img * (n - 1)
    i0 = np.floor(pos).astype(int)
    d = pos - i0
    i1 = np.clip(i0 + 1, 0, n - 1)
    i0 = np.clip(i0, 0, n - 1)

    # gather 8 corners and blend
    def g(ix, iy, iz):
        return LUT[ix, iy, iz]

    c000 = g(i0[...,0], i0[...,1], i0[...,2])
    c100 = g(i1[...,0], i0[...,1], i0[...,2])
    c010 = g(i0[...,0], i1[...,1], i0[...,2])
    c110 = g(i1[...,0], i1[...,1], i0[...,2])
    c001 = g(i0[...,0], i0[...,1], i1[...,2])
    c101 = g(i1[...,0], i0[...,1], i1[...,2])
    c011 = g(i0[...,0], i1[...,1], i1[...,2])
    c111 = g(i1[...,0], i1[...,1], i1[...,2])

    wx, wy, wz = d[...,0:1], d[...,1:2], d[...,2:3]
    c00 = c000*(1-wx) + c100*wx
    c01 = c001*(1-wx) + c101*wx
    c10 = c010*(1-wx) + c110*wx
    c11 = c011*(1-wx) + c111*wx
    c0  = c00*(1-wy) + c10*wy
    c1  = c01*(1-wy) + c11*wy
    out = c0*(1-wz) + c1*wz
    return np.clip(out, 0, 1)

import numpy as np

def build_lut3d_from_pairs(
    src, dst, n=33, round_index=True,
    smooth_iters=24, sigma=0.9, pyramid_levels=2, keep_known=True
):
    """
    Robust LUT from scattered pairs using mask-aware normalized Gaussian diffusion
    with a coarse-to-fine (pyramidal) inpainting.

    Steps:
      1) Bin: for each pair (src -> dst), accumulate src into voxel of dst, then average.
      2) Inpaint (coarse-to-fine): propagate values into holes with mask-normalized
         separable Gaussian smoothing; refine at higher resolutions.
      3) Optional: keep original observed voxels fixed during diffusion (keep_known).

    Parameters
    ----------
    src, dst : (m,3) float in [0,1]
    n        : int, LUT edge
    round_index : if True, round indices; else floor
    smooth_iters : total smoothing iterations per level
    sigma    : base Gaussian sigma (in voxels) for each iteration (separable)
    pyramid_levels : number of downsampled levels (>=1). 1 disables pyramid.
    keep_known : if True, observed voxels are clamped during diffusion

    Returns
    -------
    lut : (n,n,n,3) float32 in [0,1]
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3

    # --- 1) Bin/average in OUTPUT space ---
    idxf = dst * (n - 1)
    idx = np.rint(idxf) if round_index else np.floor(idxf)
    idx = np.clip(idx, 0, n - 1).astype(np.int64)

    sums = np.zeros((n, n, n, 3), dtype=np.float64)
    cnts = np.zeros((n, n, n), dtype=np.int64)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    np.add.at(sums, (x, y, z, slice(None)), src)
    np.add.at(cnts, (x, y, z), 1)
    lut = np.zeros_like(sums)
    mask = cnts > 0
    lut[mask] = sums[mask] / cnts[mask, None]

    # --- 2) Coarse-to-fine inpainting with normalized Gaussian diffusion ---
    lut = _inpaint_pyramid(lut, mask, levels=pyramid_levels,
                           iters=smooth_iters, sigma=sigma, keep_known=keep_known)

    return np.clip(lut, 0.0, 1.0).astype(np.float32)


# ---------------- core diffusion tools ----------------

def _inpaint_pyramid(lut, mask, levels=2, iters=24, sigma=0.9, keep_known=True):
    """
    Build a pyramid, fill at coarse level, then upsample and refine at each level
    via mask-normalized Gaussian smoothing (separable). Much smoother than plain
    neighbor averaging and robust on sparse masks.
    """
    levels = max(1, int(levels))
    # Build pyramid (coarsen by factor 2 each level using masked average)
    pyr_vals = [lut]
    pyr_mask = [mask]
    for _ in range(1, levels):
        v2, m2 = _downsample2_masked(pyr_vals[-1], pyr_mask[-1])
        pyr_vals.append(v2)
        pyr_mask.append(m2)

    # Coarsest level: initialize holes to global mean for stability
    v = pyr_vals[-1].copy()
    m = pyr_mask[-1].copy()
    if not m.all():
        mean_val = v[m].reshape(-1, 3).mean(axis=0) if m.any() else np.array([0.5, 0.5, 0.5])
        v[~m] = mean_val

    # Diffuse at coarsest
    v = _diffuse_gaussian_normalized(v, m, iters=iters, sigma=sigma, keep_known=keep_known)

    # Go up the pyramid
    for level in range(levels - 2, -1, -1):
        v = _upsample2_trilinear(v, target_shape=pyr_vals[level].shape[:3])
        m = pyr_mask[level]
        # If keep_known, re-impose observed voxels before refinement
        if keep_known:
            v[m] = pyr_vals[level][m]
        else:
            # Blend gently to avoid seams
            v = np.where(m[..., None], pyr_vals[level], v)
        v = _diffuse_gaussian_normalized(v, m, iters=iters, sigma=sigma, keep_known=keep_known)

    return v


def _gaussian_kernel_1d(sigma):
    # small, stable kernel; clamp radius at 3*sigma
    sigma = max(1e-6, float(sigma))
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k  # shape (2r+1,)


def _separable_gaussian_normalized(values, mask, sigma):
    """
    Mask-aware normalized convolution with separable Gaussian.
    values: (n,n,n,3), mask: (n,n,n) boolean
    """
    k = _gaussian_kernel_1d(sigma)
    # weights are the mask convolved with Gaussian (3D separable)
    w = mask.astype(np.float64)
    w = _conv1d_axis(w, k, axis=0)
    w = _conv1d_axis(w, k, axis=1)
    w = _conv1d_axis(w, k, axis=2)

    # convolve values pre-multiplied by mask
    v = values * mask[..., None]
    for ax in (0, 1, 2):
        v = _conv1d_axis(v, k, axis=ax)

    # normalize safely
    eps = 1e-12
    out = v / np.maximum(w[..., None], eps)
    return out, w


def _diffuse_gaussian_normalized(vals, mask, iters=24, sigma=0.9, keep_known=True):
    """
    Iteratively apply mask-normalized Gaussian smoothing.
    If keep_known=True, clamp observed voxels every iteration (preserves anchors).
    """
    v = vals.copy()
    for _ in range(max(1, int(iters))):
        smoothed, _ = _separable_gaussian_normalized(v, mask, sigma)
        if keep_known:
            v[~mask] = smoothed[~mask]
        else:
            v = smoothed
    return v


def _conv1d_axis(arr, kernel, axis):
    """
    1D convolution along `axis` with symmetric padding.
    arr: (..., C) or (...) array, kernel: (K,)
    """
    K = kernel.shape[0]
    r = (K - 1) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (r, r)
    padded = np.pad(arr, pad_width, mode='edge')
    # roll-and-weight (efficient for small kernels and small grids)
    out = np.zeros_like(arr, dtype=np.float64)
    for i, w in enumerate(kernel):
        shift = i - r
        out += w * np.take(padded, indices=range(r + shift, r + shift + arr.shape[axis]), axis=axis)
    return out


def _downsample2_masked(vals, mask):
    """
    Downsample by factor 2 with mask-aware average.
    """
    n, m, l, c = vals.shape
    n2, m2, l2 = (max(1, n // 2), max(1, m // 2), max(1, l // 2))

    # slice pairs
    v = vals[:2 * n2, :2 * m2, :2 * l2]
    mk = mask[:2 * n2, :2 * m2, :2 * l2]

    # sum over 2x2x2 blocks
    v_sum = (
        v[0::2, 0::2, 0::2] + v[1::2, 0::2, 0::2] +
        v[0::2, 1::2, 0::2] + v[1::2, 1::2, 0::2] +
        v[0::2, 0::2, 1::2] + v[1::2, 0::2, 1::2] +
        v[0::2, 1::2, 1::2] + v[1::2, 1::2, 1::2]
    )
    w_sum = (
        mk[0::2, 0::2, 0::2] + mk[1::2, 0::2, 0::2] +
        mk[0::2, 1::2, 0::2] + mk[1::2, 1::2, 0::2] +
        mk[0::2, 0::2, 1::2] + mk[1::2, 0::2, 1::2] +
        mk[0::2, 1::2, 1::2] + mk[1::2, 1::2, 1::2]
    ).astype(np.float64)

    eps = 1e-12
    out = v_sum / np.maximum(w_sum[..., None], eps)
    out[w_sum == 0] = 0.5  # neutral fallback if a block had no known voxels
    m2 = w_sum > 0
    return out, m2


def _upsample2_trilinear(vals, target_shape):
    """
    Upsample by factor ~2 (to `target_shape`) using trilinear interpolation.
    """
    n, m, l, c = vals.shape
    tn, tm, tl = target_shape
    # normalized coords in source grid
    gx = np.linspace(0, n - 1, tn)
    gy = np.linspace(0, m - 1, tm)
    gz = np.linspace(0, l - 1, tl)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
    x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int); z0 = np.floor(Z).astype(int)
    x1 = np.clip(x0 + 1, 0, n - 1); y1 = np.clip(y0 + 1, 0, m - 1); z1 = np.clip(z0 + 1, 0, l - 1)
    tx = (X - x0)[..., None]; ty = (Y - y0)[..., None]; tz = (Z - z0)[..., None]

    def g(ix, iy, iz): return vals[ix, iy, iz]
    c000 = g(x0, y0, z0); c100 = g(x1, y0, z0)
    c010 = g(x0, y1, z0); c110 = g(x1, y1, z0)
    c001 = g(x0, y0, z1); c101 = g(x1, y0, z1)
    c011 = g(x0, y1, z1); c111 = g(x1, y1, z1)
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    out = c0 * (1 - tz) + c1 * tz
    return out


import numpy as np

def build_lut3d_from_pairs_laplacian(src, dst, n=33, round_index=True,
                                     max_iters=40000, tol=1e-5):
    """
    Build a 3D LUT (n,n,n,3) by binning (src->dst) pairs in output space,
    then fill holes with classical Laplacian (harmonic) diffusion.

    Steps
      1) Bin: accumulate 'src' into voxel at rounded (dst*(n-1)); average.
      2) Inpaint: solve ∇²U = 0 on unknown voxels with Dirichlet boundary
         on known voxels via Jacobi iterations (6-neighbor averaging).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3

    # --- bin/average in output space ---
    idxf = dst * (n - 1)
    idx = np.rint(idxf) if round_index else np.floor(idxf)
    idx = np.clip(idx, 0, n - 1).astype(np.int64)

    sums = np.zeros((n, n, n, 3), dtype=np.float64)
    cnts = np.zeros((n, n, n), dtype=np.int64)

    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    np.add.at(sums, (x, y, z, slice(None)), src)
    np.add.at(cnts, (x, y, z), 1)

    lut = np.zeros_like(sums)
    mask = cnts > 0  # True where we have observed data
    lut[mask] = sums[mask] / cnts[mask, None]

    # --- Laplacian diffusion to fill holes ---
    lut = laplacian_inpaint_3d(lut, mask, max_iters=max_iters, tol=tol)

    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def laplacian_inpaint_3d(values, mask, max_iters=4000, tol=1e-5):
    """
    Classical Laplacian diffusion (Jacobi) on a 3D vector field:
      For unknown voxels: U_new = mean(6-neighbors of U)
      Known voxels remain clamped (Dirichlet boundary inside the volume).

    values: (n,n,n,3) float64, initial (knowns filled, unknowns arbitrary)
    mask  : (n,n,n) bool, True on known voxels
    """
    v = values.copy()
    known = mask.astype(bool)
    unknown = ~known
    if not unknown.any():
        return v

    # Precompute neighbor count per voxel (handles borders: fewer than 6)
    ones = np.ones(known.shape, dtype=np.float64)
    neigh_count = (
        np.roll(ones,  1, axis=0) + np.roll(ones, -1, axis=0) +
        np.roll(ones,  1, axis=1) + np.roll(ones, -1, axis=1) +
        np.roll(ones,  1, axis=2) + np.roll(ones, -1, axis=2)
    )

    # Jacobi iterations
    for it in range(int(max_iters)):
        # Sum of neighbors (vector-valued)
        s = (
            np.roll(v,  1, axis=0) + np.roll(v, -1, axis=0) +
            np.roll(v,  1, axis=1) + np.roll(v, -1, axis=1) +
            np.roll(v,  1, axis=2) + np.roll(v, -1, axis=2)
        )
        v_new = s / neigh_count[..., None]

        # clamp known voxels to their original values
        v_new[known] = values[known]

        # convergence check on unknowns
        delta = np.max(np.abs(v_new[unknown] - v[unknown])) if unknown.any() else 0.0
        v = v_new
        if delta < tol:
            break

    return v

def modiftime():
    return int(os.path.getmtime(__file__))