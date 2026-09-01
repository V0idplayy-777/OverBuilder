// Hand-written WebAssembly text (WAT) kernels.
// Two cache/register-tiled SGEMM variants covering every GEMM shape the
// transformer needs:
//
//   sgemm_nt: C[M,N] = A[M,K] * B[N,K]^T   (row-dot kernel, both operand rows contiguous)
//             used for: forward linear layers (Y = X W^T), attn scores Q K^T, dP = dY V^T
//
//   sgemm_nn: C[M,N] = A[M,K] * B[K,N]     (ikj kernel, B row contiguous)
//             used for: weight/input gradients, context = P V, dQ/dK/dV
//
// Register tiling: 2 rows x 4 cols of C accumulated in f32 locals.
// Requirements (asserted by the JS dispatcher):
//   M % 2 == 0, N % 4 == 0, K % 2 == 0 for nt; K >= 1 for nn.
//
// `beta` is 0 or 1: C = A*B (beta 0, plain store) or C += A*B (beta 1).

export const WAT_SOURCE = `
(module
  (memory (export "memory") 96 16384)

  ;; ---------------------------------------------------------------- nt
  (func (export "sgemm_nt")
    (param $pa i32) (param $pb i32) (param $pc i32)
    (param $M i32) (param $N i32) (param $K i32) (param $beta i32)
    (local $i i32) (local $j i32) (local $k i32)
    (local $pA0 i32) (local $pA1 i32)
    (local $pB0 i32) (local $pB1 i32) (local $pB2 i32) (local $pB3 i32)
    (local $pC i32)
    (local $K4 i32) (local $N4 i32) (local $bf f32)
    (local $a0 f32) (local $a1 f32) (local $b0 f32) (local $b1 f32)
    (local $t0 f32) (local $t1 f32)
    (local $r00 f32) (local $r01 f32) (local $r02 f32) (local $r03 f32)
    (local $r10 f32) (local $r11 f32) (local $r12 f32) (local $r13 f32)
    (local.set $bf (f32.convert_i32_s (local.get $beta)))
    (local.set $K4 (i32.shl (local.get $K) (i32.const 2)))
    (local.set $N4 (i32.shl (local.get $N) (i32.const 2)))
    (local.set $i (i32.const 0))
    (loop $li
      (local.set $j (i32.const 0))
      (loop $lj
        (local.set $pA0 (i32.add (local.get $pa) (i32.mul (local.get $i) (local.get $K4))))
        (local.set $pA1 (i32.add (local.get $pA0) (local.get $K4)))
        ;; pC = c + (i*N + j)*4
        (local.set $pC (i32.add (local.get $pc)
          (i32.shl (i32.add (i32.mul (local.get $i) (local.get $N)) (local.get $j)) (i32.const 2))))
        ;; init accumulators with beta * C
        (local.set $r00 (f32.mul (local.get $bf) (f32.load (local.get $pC))))
        (local.set $r01 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.const 4)))))
        (local.set $r02 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.const 8)))))
        (local.set $r03 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.const 12)))))
        (local.set $r10 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (local.get $N4)))))
        (local.set $r11 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 4))))))
        (local.set $r12 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 8))))))
        (local.set $r13 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 12))))))
        ;; B row pointers: b + (j+t)*K*4
        (local.set $pB0 (i32.add (local.get $pb) (i32.mul (local.get $j) (local.get $K4))))
        (local.set $pB1 (i32.add (local.get $pB0) (local.get $K4)))
        (local.set $pB2 (i32.add (local.get $pB1) (local.get $K4)))
        (local.set $pB3 (i32.add (local.get $pB2) (local.get $K4)))
        ;; k loop, unrolled by 2
        (local.set $k (i32.const 0))
        (loop $lk
          (local.set $a0 (f32.load (local.get $pA0)))
          (local.set $a1 (f32.load (i32.add (local.get $pA0) (i32.const 4))))
          (local.set $b0 (f32.load (local.get $pA1)))
          (local.set $b1 (f32.load (i32.add (local.get $pA1) (i32.const 4))))
          ;; B row 0
          (local.set $t0 (f32.load (local.get $pB0)))
          (local.set $t1 (f32.load (i32.add (local.get $pB0) (i32.const 4))))
          (local.set $r00 (f32.add (local.get $r00) (f32.add (f32.mul (local.get $a0) (local.get $t0)) (f32.mul (local.get $a1) (local.get $t1)))))
          (local.set $r10 (f32.add (local.get $r10) (f32.add (f32.mul (local.get $b0) (local.get $t0)) (f32.mul (local.get $b1) (local.get $t1)))))
          ;; B row 1
          (local.set $t0 (f32.load (local.get $pB1)))
          (local.set $t1 (f32.load (i32.add (local.get $pB1) (i32.const 4))))
          (local.set $r01 (f32.add (local.get $r01) (f32.add (f32.mul (local.get $a0) (local.get $t0)) (f32.mul (local.get $a1) (local.get $t1)))))
          (local.set $r11 (f32.add (local.get $r11) (f32.add (f32.mul (local.get $b0) (local.get $t0)) (f32.mul (local.get $b1) (local.get $t1)))))
          ;; B row 2
          (local.set $t0 (f32.load (local.get $pB2)))
          (local.set $t1 (f32.load (i32.add (local.get $pB2) (i32.const 4))))
          (local.set $r02 (f32.add (local.get $r02) (f32.add (f32.mul (local.get $a0) (local.get $t0)) (f32.mul (local.get $a1) (local.get $t1)))))
          (local.set $r12 (f32.add (local.get $r12) (f32.add (f32.mul (local.get $b0) (local.get $t0)) (f32.mul (local.get $b1) (local.get $t1)))))
          ;; B row 3
          (local.set $t0 (f32.load (local.get $pB3)))
          (local.set $t1 (f32.load (i32.add (local.get $pB3) (i32.const 4))))
          (local.set $r03 (f32.add (local.get $r03) (f32.add (f32.mul (local.get $a0) (local.get $t0)) (f32.mul (local.get $a1) (local.get $t1)))))
          (local.set $r13 (f32.add (local.get $r13) (f32.add (f32.mul (local.get $b0) (local.get $t0)) (f32.mul (local.get $b1) (local.get $t1)))))
          ;; advance pointers by 8 bytes (2 floats)
          (local.set $pA0 (i32.add (local.get $pA0) (i32.const 8)))
          (local.set $pA1 (i32.add (local.get $pA1) (i32.const 8)))
          (local.set $pB0 (i32.add (local.get $pB0) (i32.const 8)))
          (local.set $pB1 (i32.add (local.get $pB1) (i32.const 8)))
          (local.set $pB2 (i32.add (local.get $pB2) (i32.const 8)))
          (local.set $pB3 (i32.add (local.get $pB3) (i32.const 8)))
          (local.set $k (i32.add (local.get $k) (i32.const 2)))
          (br_if $lk (i32.lt_u (local.get $k) (local.get $K)))
        )
        ;; store 2x4 tile
        (f32.store (local.get $pC) (local.get $r00))
        (f32.store (i32.add (local.get $pC) (i32.const 4)) (local.get $r01))
        (f32.store (i32.add (local.get $pC) (i32.const 8)) (local.get $r02))
        (f32.store (i32.add (local.get $pC) (i32.const 12)) (local.get $r03))
        (f32.store (i32.add (local.get $pC) (local.get $N4)) (local.get $r10))
        (f32.store (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 4))) (local.get $r11))
        (f32.store (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 8))) (local.get $r12))
        (f32.store (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 12))) (local.get $r13))
        (local.set $j (i32.add (local.get $j) (i32.const 4)))
        (br_if $lj (i32.lt_u (local.get $j) (local.get $N)))
      )
      (local.set $i (i32.add (local.get $i) (i32.const 2)))
      (br_if $li (i32.lt_u (local.get $i) (local.get $M)))
    )
  )

  ;; ---------------------------------------------------------------- nn
  (func (export "sgemm_nn")
    (param $pa i32) (param $pb i32) (param $pc i32)
    (param $M i32) (param $N i32) (param $K i32) (param $beta i32)
    (local $i i32) (local $j i32) (local $k i32)
    (local $pA0 i32) (local $pA1 i32) (local $pB i32) (local $pC i32)
    (local $K4 i32) (local $N4 i32) (local $bf f32)
    (local $a0 f32) (local $a1 f32)
    (local $t0 f32) (local $t1 f32) (local $t2 f32) (local $t3 f32)
    (local $r00 f32) (local $r01 f32) (local $r02 f32) (local $r03 f32)
    (local $r10 f32) (local $r11 f32) (local $r12 f32) (local $r13 f32)
    (local.set $bf (f32.convert_i32_s (local.get $beta)))
    (local.set $K4 (i32.shl (local.get $K) (i32.const 2)))
    (local.set $N4 (i32.shl (local.get $N) (i32.const 2)))
    (local.set $i (i32.const 0))
    (loop $li
      (local.set $pA0 (i32.add (local.get $pa) (i32.mul (local.get $i) (local.get $K4))))
      (local.set $pA1 (i32.add (local.get $pA0) (local.get $K4)))
      (local.set $j (i32.const 0))
      (loop $lj
        (local.set $pC (i32.add (local.get $pc)
          (i32.shl (i32.add (i32.mul (local.get $i) (local.get $N)) (local.get $j)) (i32.const 2))))
        (local.set $r00 (f32.mul (local.get $bf) (f32.load (local.get $pC))))
        (local.set $r01 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.const 4)))))
        (local.set $r02 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.const 8)))))
        (local.set $r03 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.const 12)))))
        (local.set $r10 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (local.get $N4)))))
        (local.set $r11 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 4))))))
        (local.set $r12 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 8))))))
        (local.set $r13 (f32.mul (local.get $bf) (f32.load (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 12))))))
        ;; B pointer starts at row k=0, col j
        (local.set $pB (i32.add (local.get $pb) (i32.shl (local.get $j) (i32.const 2))))
        ;; save A pointers at loop start (restore per k via separate locals)
        (local.set $k (i32.const 0))
        (loop $lk
          (local.set $a0 (f32.load (i32.add (local.get $pA0) (i32.shl (local.get $k) (i32.const 2)))))
          (local.set $a1 (f32.load (i32.add (local.get $pA1) (i32.shl (local.get $k) (i32.const 2)))))
          (local.set $t0 (f32.load (local.get $pB)))
          (local.set $t1 (f32.load (i32.add (local.get $pB) (i32.const 4))))
          (local.set $t2 (f32.load (i32.add (local.get $pB) (i32.const 8))))
          (local.set $t3 (f32.load (i32.add (local.get $pB) (i32.const 12))))
          (local.set $r00 (f32.add (local.get $r00) (f32.mul (local.get $a0) (local.get $t0))))
          (local.set $r01 (f32.add (local.get $r01) (f32.mul (local.get $a0) (local.get $t1))))
          (local.set $r02 (f32.add (local.get $r02) (f32.mul (local.get $a0) (local.get $t2))))
          (local.set $r03 (f32.add (local.get $r03) (f32.mul (local.get $a0) (local.get $t3))))
          (local.set $r10 (f32.add (local.get $r10) (f32.mul (local.get $a1) (local.get $t0))))
          (local.set $r11 (f32.add (local.get $r11) (f32.mul (local.get $a1) (local.get $t1))))
          (local.set $r12 (f32.add (local.get $r12) (f32.mul (local.get $a1) (local.get $t2))))
          (local.set $r13 (f32.add (local.get $r13) (f32.mul (local.get $a1) (local.get $t3))))
          (local.set $pB (i32.add (local.get $pB) (local.get $N4)))
          (local.set $k (i32.add (local.get $k) (i32.const 1)))
          (br_if $lk (i32.lt_u (local.get $k) (local.get $K)))
        )
        (f32.store (local.get $pC) (local.get $r00))
        (f32.store (i32.add (local.get $pC) (i32.const 4)) (local.get $r01))
        (f32.store (i32.add (local.get $pC) (i32.const 8)) (local.get $r02))
        (f32.store (i32.add (local.get $pC) (i32.const 12)) (local.get $r03))
        (f32.store (i32.add (local.get $pC) (local.get $N4)) (local.get $r10))
        (f32.store (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 4))) (local.get $r11))
        (f32.store (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 8))) (local.get $r12))
        (f32.store (i32.add (local.get $pC) (i32.add (local.get $N4) (i32.const 12))) (local.get $r13))
        (local.set $j (i32.add (local.get $j) (i32.const 4)))
        (br_if $lj (i32.lt_u (local.get $j) (local.get $N)))
      )
      (local.set $i (i32.add (local.get $i) (i32.const 2)))
      (br_if $li (i32.lt_u (local.get $i) (local.get $M)))
    )
  )
)
`;
