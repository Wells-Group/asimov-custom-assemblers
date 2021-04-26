template <typename T, typename E, typename V>
inline void dot34(const T& A, const E& B, V& out)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      out.unchecked(i, j)
          = A.unchecked(i, 0) * B.unchecked(0, j) + A.unchecked(i, 1) * B.unchecked(1, j)
            + A.unchecked(i, 2) * B.unchecked(2, j) + A.unchecked(i, 3) * B.unchecked(3, j);
}