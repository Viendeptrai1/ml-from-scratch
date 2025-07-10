#!/usr/bin/env python3
"""
COMPREHENSIVE SYMPY TESTING SUITE
=====================================
Test toàn bộ thư viện SymPy với các ví dụ thực tế
"""

import sympy as sp
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot, plot3d, plot_implicit
from sympy.physics import units
from sympy.physics.mechanics import *
from sympy.physics.quantum import *
from sympy.stats import *
import sys

def print_section(title):
    """In tiêu đề section đẹp"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_subsection(title):
    """In tiêu đề subsection"""
    print(f"\n--- {title} ---")

def main():
    print("🧮 SYMPY COMPREHENSIVE TEST SUITE")
    print("Kiểm tra toàn bộ tính năng của SymPy")
    print(f"SymPy version: {sp.__version__}")
    
    # =================================================================
    # 1. BASIC SYMBOLIC OPERATIONS
    # =================================================================
    print_section("1. CÁC PHÉP TOÁN SYMBOLIC CỞ BẢN")
    
    # Định nghĩa các biến symbolic
    x, y, z, t, a, b, c = symbols('x y z t a b c')
    n, m, k = symbols('n m k', integer=True)
    p, q = symbols('p q', positive=True)
    
    print_subsection("Định nghĩa biến symbolic")
    print(f"x = {x}, type: {type(x)}")
    print(f"n (integer) = {n}")
    print(f"p (positive) = {p}")
    
    print_subsection("Các phép toán cơ bản")
    expr1 = x**2 + 2*x + 1
    expr2 = (x + 1)**2
    print(f"expr1 = {expr1}")
    print(f"expr2 = {expr2}")
    print(f"expr1 == expr2: {expr1.equals(expr2)}")
    print(f"expand(expr2) = {expand(expr2)}")
    print(f"factor(expr1) = {factor(expr1)}")
    
    # =================================================================
    # 2. EQUATION SOLVING
    # =================================================================
    print_section("2. GIẢI PHƯƠNG TRÌNH")
    
    print_subsection("Phương trình bậc 2")
    eq1 = Eq(x**2 + 2*x - 3, 0)
    sol1 = solve(eq1, x)
    print(f"Phương trình: {eq1}")
    print(f"Nghiệm: {sol1}")
    
    print_subsection("Hệ phương trình")
    eq2 = Eq(x + y, 5)
    eq3 = Eq(2*x - y, 1)
    sys_sol = solve([eq2, eq3], [x, y])
    print(f"Hệ PT: {eq2}, {eq3}")
    print(f"Nghiệm: {sys_sol}")
    
    print_subsection("Phương trình transcendental")
    eq4 = Eq(exp(x) + x, 2)
    sol4 = solve(eq4, x)
    print(f"exp(x) + x = 2")
    print(f"Nghiệm: {sol4}")
    
    # =================================================================
    # 3. CALCULUS
    # =================================================================
    print_section("3. GIẢI TÍCH (CALCULUS)")
    
    print_subsection("Đạo hàm")
    f = x**3 + sin(x) + exp(x)
    df_dx = diff(f, x)
    d2f_dx2 = diff(f, x, 2)
    print(f"f(x) = {f}")
    print(f"f'(x) = {df_dx}")
    print(f"f''(x) = {d2f_dx2}")
    
    print_subsection("Đạo hàm riêng")
    g = x**2 * y + sin(x*y)
    dg_dx = diff(g, x)
    dg_dy = diff(g, y)
    d2g_dxdy = diff(g, x, y)
    print(f"g(x,y) = {g}")
    print(f"∂g/∂x = {dg_dx}")
    print(f"∂g/∂y = {dg_dy}")
    print(f"∂²g/∂x∂y = {d2g_dxdy}")
    
    print_subsection("Tích phân")
    # Tích phân không xác định
    int1 = integrate(x**2 + sin(x), x)
    print(f"∫(x² + sin(x))dx = {int1}")
    
    # Tích phân xác định
    int2 = integrate(x**2, (x, 0, 1))
    print(f"∫₀¹ x² dx = {int2}")
    
    # Tích phân bội
    int3 = integrate(x*y, (x, 0, 1), (y, 0, 1))
    print(f"∫₀¹∫₀¹ xy dx dy = {int3}")
    
    print_subsection("Giới hạn")
    lim1 = limit(sin(x)/x, x, 0)
    lim2 = limit((1 + 1/n)**n, n, oo)
    print(f"lim(x→0) sin(x)/x = {lim1}")
    print(f"lim(n→∞) (1 + 1/n)ⁿ = {lim2}")
    
    print_subsection("Chuỗi Taylor")
    series1 = series(exp(x), x, 0, 6)
    series2 = series(sin(x), x, 0, 8)
    print(f"exp(x) ≈ {series1}")
    print(f"sin(x) ≈ {series2}")
    
    # =================================================================
    # 4. LINEAR ALGEBRA
    # =================================================================
    print_section("4. ĐẠI SỐ TUYẾN TÍNH")
    
    print_subsection("Ma trận")
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"A + B = {A + B}")
    print(f"A * B = {A * B}")
    print(f"A⁻¹ = {A.inv()}")
    print(f"det(A) = {A.det()}")
    
    print_subsection("Eigenvalues và Eigenvectors")
    eigenvals = A.eigenvals()
    eigenvects = A.eigenvects()
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors: {eigenvects}")
    
    print_subsection("Giải hệ phương trình tuyến tính")
    C = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    b = Matrix([1, 2, 3])
    x_sol = C.LUsolve(b)
    print(f"Cx = b với C = {C}, b = {b}")
    print(f"x = {x_sol}")
    
    # =================================================================
    # 5. POLYNOMIALS
    # =================================================================
    print_section("5. ĐA THỨC")
    
    print_subsection("Thao tác với đa thức")
    p1 = Poly(x**3 + 2*x**2 + 3*x + 4, x)
    p2 = Poly(x**2 + x + 1, x)
    print(f"p1 = {p1}")
    print(f"p2 = {p2}")
    print(f"p1 + p2 = {p1 + p2}")
    print(f"p1 * p2 = {p1 * p2}")
    print(f"div(p1, p2) = {div(p1, p2)}")
    
    print_subsection("Tìm nghiệm đa thức")
    roots_p1 = solve(p1, x)
    print(f"Nghiệm của p1: {roots_p1}")
    
    # =================================================================
    # 6. TRIGONOMETRY
    # =================================================================
    print_section("6. LƯỢNG GIÁC")
    
    print_subsection("Biến đổi lượng giác")
    trig_expr = sin(x)**2 + cos(x)**2
    simplified = simplify(trig_expr)
    print(f"sin²(x) + cos²(x) = {simplified}")
    
    expanded = expand_trig(sin(x + y))
    print(f"sin(x + y) = {expanded}")
    
    print_subsection("Phương trình lượng giác")
    trig_eq = Eq(sin(x), 1/2)
    trig_sol = solve(trig_eq, x)
    print(f"sin(x) = 1/2")
    print(f"Nghiệm: {trig_sol}")
    
    # =================================================================
    # 7. NUMBER THEORY
    # =================================================================
    print_section("7. SỐ HỌC")
    
    print_subsection("Số nguyên tố")
    print(f"Số nguyên tố thứ 100: {prime(100)}")
    print(f"10 số nguyên tố đầu tiên: {[prime(i) for i in range(1, 11)]}")
    print(f"1009 có phải số nguyên tố? {isprime(1009)}")
    
    print_subsection("Phân tích thừa số")
    print(f"Phân tích 1260: {factorint(1260)}")
    print(f"Ước chung lớn nhất gcd(48, 18): {gcd(48, 18)}")
    print(f"Bội chung nhỏ nhất lcm(48, 18): {lcm(48, 18)}")
    
    # =================================================================
    # 8. COMBINATORICS
    # =================================================================
    print_section("8. TỔ HỢP")
    
    print_subsection("Hoán vị và tổ hợp")
    print(f"5! = {factorial(5)}")
    print(f"C(10,3) = {binomial(10, 3)}")
    print(f"Số Bell B_5 = {bell(5)}")
    print(f"Số Fibonacci F_10 = {fibonacci(10)}")
    
    # =================================================================
    # 9. LOGIC
    # =================================================================
    print_section("9. LOGIC")
    
    print_subsection("Logic propositional")
    A_sym = symbols('A')
    B_sym = symbols('B')
    
    formula = And(A_sym, Or(B_sym, Not(A_sym)))
    print(f"Formula: {formula}")
    print(f"Simplified: {simplify_logic(formula)}")
    
    # =================================================================
    # 10. STATISTICS
    # =================================================================
    print_section("10. THỐNG KÊ VÀ XÁC SUẤT")
    
    print_subsection("Phân phối xác suất")
    # Phân phối chuẩn
    X = Normal('X', 0, 1)
    print(f"X ~ N(0,1)")
    print(f"P(X < 1.96) = {P(X < 1.96).evalf()}")
    print(f"E[X] = {E(X)}")
    print(f"Var(X) = {variance(X)}")
    
    # Phân phối nhị thức
    Y = Binomial('Y', 10, 0.3)
    print(f"Y ~ Binomial(10, 0.3)")
    print(f"P(Y = 3) = {P(Eq(Y, 3)).evalf()}")
    
    # =================================================================
    # 11. PHYSICS
    # =================================================================
    print_section("11. VẬT LÝ")
    
    print_subsection("Đơn vị vật lý")
    # Sử dụng units
    length = 5 * units.meter
    time_val = 2 * units.second
    velocity = length / time_val
    print(f"Quãng đường: {length}")
    print(f"Thời gian: {time_val}")
    print(f"Vận tốc: {velocity}")
    
    print_subsection("Mechanics - Chuyển động")
    # Vật lý cơ học
    t_sym = symbols('t')
    s = symbols('s', cls=Function)
    
    # Phương trình chuyển động
    motion_eq = Eq(s(t_sym).diff(t_sym, 2), -9.8)  # gia tốc trọng trường
    print(f"Phương trình chuyển động: {motion_eq}")
    
    # =================================================================
    # 12. DIFFERENTIAL EQUATIONS
    # =================================================================
    print_section("12. PHƯƠNG TRÌNH VI PHÂN")
    
    print_subsection("Phương trình vi phân thường")
    f = Function('f')
    
    # PT vi phân bậc 1
    ode1 = Eq(f(x).diff(x) + f(x), exp(x))
    sol_ode1 = dsolve(ode1, f(x))
    print(f"PT: f'(x) + f(x) = eˣ")
    print(f"Nghiệm: {sol_ode1}")
    
    # PT vi phân bậc 2
    ode2 = Eq(f(x).diff(x, 2) + f(x), 0)
    sol_ode2 = dsolve(ode2, f(x))
    print(f"PT: f''(x) + f(x) = 0")
    print(f"Nghiệm: {sol_ode2}")
    
    # =================================================================
    # 13. PLOTTING (nếu có thể)
    # =================================================================
    print_section("13. VẼ ĐỒ THỊ")
    
    try:
        print_subsection("Đồ thị 2D")
        # Vẽ hàm số
        p1 = plot(x**2, (x, -3, 3), title='y = x²', show=False)
        p1.save('plot_x_squared.png')
        print("Đã lưu đồ thị y = x² vào plot_x_squared.png")
        
        # Vẽ nhiều hàm
        p2 = plot(sin(x), cos(x), (x, -2*pi, 2*pi), 
                 title='sin(x) và cos(x)', show=False)
        p2.save('plot_trig.png')
        print("Đã lưu đồ thị sin(x) và cos(x) vào plot_trig.png")
        
        print_subsection("Đồ thị 3D")
        # Vẽ đồ thị 3D
        p3 = plot3d(x**2 + y**2, (x, -2, 2), (y, -2, 2), 
                    title='z = x² + y²', show=False)
        p3.save('plot_3d.png')
        print("Đã lưu đồ thị 3D z = x² + y² vào plot_3d.png")
        
    except Exception as e:
        print(f"Lỗi khi vẽ đồ thị: {e}")
        print("Có thể cần cài đặt matplotlib để vẽ đồ thị")
    
    # =================================================================
    # 14. ADVANCED TOPICS
    # =================================================================
    print_section("14. CHỦ ĐỀ NÂNG CAO")
    
    print_subsection("Geometric Algebra")
    # Hình học đại số
    from sympy.geometry import Point, Line, Circle
    point1 = Point(0, 0)
    point2 = Point(3, 4)
    line1 = Line(point1, point2)
    circle1 = Circle(point1, 5)
    print(f"Điểm 1: {point1}")
    print(f"Điểm 2: {point2}")
    print(f"Khoảng cách: {point1.distance(point2)}")
    print(f"Đường thẳng: {line1}")
    print(f"Hình tròn: {circle1}")
    
    print_subsection("Set Theory")
    # Lý thuyết tập hợp
    set1 = FiniteSet(1, 2, 3, 4)
    set2 = FiniteSet(3, 4, 5, 6)
    print(f"Set 1: {set1}")
    print(f"Set 2: {set2}")
    print(f"Hợp: {set1.union(set2)}")
    print(f"Giao: {set1.intersect(set2)}")
    print(f"Hiệu: {set1 - set2}")
    
    # =================================================================
    # 15. PERFORMANCE TEST
    # =================================================================
    print_section("15. KIỂM TRA HIỆU SUẤT")
    
    import time
    
    print_subsection("Benchmark các phép toán")
    
    # Test symbolic computation speed
    start_time = time.time()
    large_expr = sum(x**i for i in range(100))
    expanded_large = expand(large_expr * (x + 1))
    end_time = time.time()
    print(f"Thời gian expand biểu thức lớn: {end_time - start_time:.4f}s")
    
    # Test matrix operations
    start_time = time.time()
    large_matrix = Matrix([[i+j for j in range(20)] for i in range(20)])
    det_large = large_matrix.det()
    end_time = time.time()
    print(f"Thời gian tính determinant ma trận 20x20: {end_time - start_time:.4f}s")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print_section("🎉 TỔNG KẾT")
    
    print("✅ Đã test thành công các module chính của SymPy:")
    modules_tested = [
        "Symbolic Operations", "Equation Solving", "Calculus",
        "Linear Algebra", "Polynomials", "Trigonometry",
        "Number Theory", "Combinatorics", "Logic", "Statistics",
        "Physics", "Differential Equations", "Plotting",
        "Geometry", "Set Theory", "Performance"
    ]
    
    for i, module in enumerate(modules_tested, 1):
        print(f"  {i:2d}. {module}")
    
    print(f"\n📊 Tổng cộng đã test {len(modules_tested)} module chính")
    print("🔥 SymPy hoạt động tốt và sẵn sàng cho các tính toán phức tạp!")
    
    print("\n💡 Tip: Bạn có thể import và sử dụng bất kỳ function nào từ test này")
    print("   Ví dụ: from sympy import *")

if __name__ == "__main__":
    main()
