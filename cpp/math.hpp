template <typename T, typename E, typename V>
inline void dot34(const T& A, const E& B, V& out)
{
  out(0, 0) = A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0);
  out(0, 1) = A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1);
  out(0, 2) = A(0, 0) * B(0, 2) + A(0, 1) * B(1, 2);
  out(1, 0) = A(1, 0) * B(0, 0) + A(1, 2) * B(2, 0);
  out(1, 1) = A(1, 0) * B(0, 1) + A(1, 2) * B(2, 1);
  out(1, 2) = A(1, 0) * B(0, 2) + A(1, 2) * B(2, 2);
  out(2, 0) = A(2, 0) * B(0, 0) + A(2, 3) * B(3, 0);
  out(2, 1) = A(2, 0) * B(0, 1) + A(2, 3) * B(3, 1);
  out(2, 2) = A(2, 0) * B(0, 2) + A(2, 3) * B(3, 2);
}