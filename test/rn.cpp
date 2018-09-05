#include <iostream>
#include <random>
#include <algorithm>

int main(){
  /* 隨機設備 */
  std::random_device rd;

  /* 隨機亂數的範圍 */
  std::cout << "Min = " << rd.min()
          << ", Max = " << rd.max() << std::endl;

  /* 產生隨機的亂數 */
  std::cout << "Random Number = " << (double) rd() / rd.max()  << std::endl;

  /* 隨機設備的熵值 */
//  std::cout << "Entropy = " << rd.entropy() << std::endl;
//	std::cout << std::min(0.1,2.0) << std::endl;
//	std::cout << std::max(1.0,2.0) << std::endl;
  return 0;
}
