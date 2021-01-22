#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <Eigen/Dense>
#include <thread>
#include <vector>
#include <cmath>
#include <sstream>
#include <iostream>
#include <time.hpp>
#include <functional>
#include "jet.hpp"
constexpr double c = 299792458;
constexpr double mu_0 = 4 * M_PI * 1e-7;
constexpr double eps_0 = 1.0 / (mu_0 * c * c);
using scalar = float;
using vector2 = Eigen::Matrix<scalar, 2, 1>;
using array  = Eigen::Array<scalar , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using array2 = Eigen::Array<vector2, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
sf::Font font;
template<typename T>
auto square(const T& arg){
    return arg * arg;
}
struct button{
    std::function<void(void)> callback;
    sf::RectangleShape backrect;
    sf::Text text;
    sf::Vector2f pos;
    sf::Vector2f size;
    button(std::function<void(void)> cb, const std::string& tx, float _x, float _y, float _xs, float _ys) : callback(cb), text(tx, font), pos(_x, _y), size(_xs, _ys){
        backrect.setPosition(sf::Vector2f(_x, _y));
        backrect.setSize(size);
        text.setPosition(sf::Vector2f(_x, _y));
        text.setFillColor(sf::Color::Green);
    }
    void on_screen_click(sf::RenderWindow& win){
        if(sf::Mouse::getPosition(win).x > pos.x && sf::Mouse::getPosition(win).x < pos.x + size.x){
            if(sf::Mouse::getPosition(win).y > pos.y && sf::Mouse::getPosition(win).y < pos.y + size.y){
                callback();
            }
        }
    }
    void draw(sf::RenderWindow& win){
        backrect.setFillColor(sf::Color(120,0,0));
        if(sf::Mouse::getPosition(win).x > pos.x && sf::Mouse::getPosition(win).x < pos.x + size.x){
            if(sf::Mouse::getPosition(win).y > pos.y && sf::Mouse::getPosition(win).y < pos.y + size.y){
                backrect.setFillColor(sf::Color(180,0,0));
            }
        }
        win.draw(backrect);
        win.draw(text);
    }
};
struct constant{
    scalar m_value;
    constant(const scalar& v) : m_value(v){}
    scalar operator()(const vector2& pos)const{
        //if(pos.y() >= 0.3 && pos.y() <= 0.4){
        //    return m_value * 3;
        //}
        return m_value;
    }
};
struct metal_slab_with_width_mu{
    scalar width;
    scalar mu_air;
    scalar mu_metal;
    metal_slab_with_width_mu(scalar w, scalar sa, scalar sm) : width(w), mu_air(sa), mu_metal(sm){}
    scalar operator()(const vector2& pos)const{
        if(pos.y() <= 1){
            return mu_air;
        }
        if(pos.y() < 1 + width){
            return mu_metal;
        }
        return mu_air;
    }
};
struct metal_slab_with_width{
    scalar width;
    scalar sigma_air;
    scalar sigma_metal;
    metal_slab_with_width(scalar w, scalar sa, scalar sm) : width(w), sigma_air(sa), sigma_metal(sm){}
    scalar operator()(const vector2& pos)const{
        if(pos.y() <= 1){
            return sigma_air;
        }
        if(pos.y() < 1 + width){
            return sigma_metal;
        }
        return sigma_air;
    }
};
struct grid_with_positions{
    size_t m, n;
    array2 positions;
    array B;
    metal_slab_with_width_mu mu;
    metal_slab_with_width sigma;
    grid_with_positions(size_t _m, size_t _n, scalar thickness) : m(_m), n(_n), positions(m, n), B(m, n), mu(thickness, mu_0, mu_0 * 100), sigma(thickness, 5, 1e7){
        size_t n4 = n / 4;
        //std::cout << n4 << "\n";
        for(size_t i = 0;i < m;i++){
            for(size_t j = 0;j < n4;j++){
                positions(i, j) = vector2(scalar(i) / scalar(m - 1), scalar(j) / scalar(n4 - 1));
            }
        }
        scalar xpos =  1.0;
        scalar width = sigma.width;
        for(size_t i = 0;i < m;i++){
            scalar inner_xpos = xpos;
            for(size_t j = n4;j < 3 * n4;j++){
                inner_xpos += width / (2 * n4);
                positions(i, j) = vector2(scalar(i) / scalar(m - 1), inner_xpos);
            }
        }
        xpos += width;
        for(size_t i = 0;i < m;i++){
            scalar inner_xpos = xpos;
            for(size_t j = 3 * n4;j < n;j++){
                inner_xpos += scalar(1) / (n4 - 1);
                positions(i, j) = vector2(scalar(i) / scalar(m - 1), inner_xpos);
            }
        }
        B.setZero();
    }
    vector2& position(size_t i, size_t j){
        return positions(i, j);
    }
    const vector2& position(size_t i, size_t j)const{
        return positions(i, j);
    }
    array interior_laplacian()const{
        array ret(m, n);
        ret.setZero();
        
        const size_t not_on_b = 1;
        //#pragma omp parallel for collapse(2) schedule(guided) num_threads(12)
        for(size_t i = 0;i < m;i++){
            for(size_t j = not_on_b;j < n;j++){
                scalar dxm1 = 0;
                scalar dxp1 = 0;
                scalar dym1 = 0;
                scalar dyp1 = 0;
                if(i > 0)
                    dxm1 = (B(i, j) - B(i - 1, j)) / square(position(i, j)(0) - position(i - 1, j)(0)) / (sigma((position(i, j) + position(i - 1, j)) / 2.0) * mu((position(i, j) + position(i - 1, j)) / 2.0));
                if(i < m - 1)
                    dxp1 = (B(i + 1, j) - B(i, j)) / square(position(i + 1, j)(0) - position(i, j)(0)) / (sigma((position(i + 1, j) + position(i, j)) / 2.0) * mu((position(i + 1, j) + position(i, j)) / 2.0));
                if(j > 0)
                    dym1 = (B(i, j) - B(i, j - 1)) / square(position(i, j)(1) - position(i, j - 1)(1)) / (sigma((position(i, j) + position(i, j - 1)) / 2.0) * mu((position(i, j) + position(i, j - 1)) / 2.0));
                if(j < n - 1)
                    dyp1 = (B(i, j + 1) - B(i, j)) / square(position(i, j + 1)(1) - position(i, j)(1)) / (sigma((position(i, j + 1) + position(i, j)) / 2.0) * mu((position(i, j + 1) + position(i, j)) / 2.0));
                ret(i, j) = (dxp1 - dxm1) + (dyp1 - dym1);
            }
        }
        return ret;
    }
};
struct double_exponential_pulse{
    double t1, t2;
    double_exponential_pulse(double _t1, double _t2) : t1(_t1), t2(_t2){}
    double operator()(double time) const{
        if(time < t1){
            return (1.0 - std::exp(-2.0 * time / t1)) / (1.0 - std::exp(-2.0));
        }
        return std::exp(-(time - t1) / (t2));
    }
};
unsigned char inc_by(unsigned char x, int by){
    for(int i = 0;i < by;i++){
        if(x < 255){
            x++;
        }
        else{
            break;
        }
    }
    return x;
}
std::string tu_string(scalar x){
    std::stringstream sstr;
    sstr << x;
    return sstr.str();
}
unsigned char fcolor(double x){
    return std::min(256 * x, 255.0);
}
void run_with_thickness(scalar thickness){
    size_t counter = 0;
    std::vector<button> buttons;
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "yeet");
    sf::Texture tex;
    double time = 0;
    bool paused = false;
    grid_with_positions grid(32, 128, thickness);
    constexpr double frequency = 10000;
    double_exponential_pulse bcond(1.0 / 4.0 / frequency, 1.0 / 4.0 / frequency);
    scalar maxf = 0;
    tex.create(grid.n, grid.m);
    sf::Sprite schprit(tex);
    schprit.setScale(window.getSize().x / (grid.n - 1.0), window.getSize().y / (grid.m - 1.0));
    window.setVerticalSyncEnabled(0);
    window.setFramerateLimit(0);
    
    font.loadFromFile("Arial.ttf");
    scalar cbmaxv = 5.0;
    sf::Vertex vertices[512];
    for(size_t i = 0;i < 256;i++){
        vertices[2 * i].position =     sf::Vector2f(800, 400 - i);
        vertices[2 * i + 1].position = sf::Vector2f(900, 400 - i);
        vertices[2 * i].color =     sf::Color(turbo_cm[i][0] * 255,turbo_cm[i][1] * 255, turbo_cm[i][2] * 255);
        vertices[2 * i + 1].color = sf::Color(turbo_cm[i][0] * 255,turbo_cm[i][1] * 255, turbo_cm[i][2] * 255);
    }
    std::string maxstr = tu_string(cbmaxv) + "Tesla";
    std::string minstr = "0 Tesla";
    
    buttons.push_back(button([&](){time = 0; grid.B.setZero();}, "Reset", 200, 10,100,40));
    buttons.push_back(button([&](){paused = !paused;}, "Pause / Unpause", 350, 10, 250,40));
    sf::Text cbmax(maxstr, font);
    sf::Text cbmin(minstr, font);
    sf::Text first_mark ("1 meter", font);
    sf::Text second_mark(tu_string(scalar(1) + grid.sigma.width) + " meters", font);
    sf::Text timet("", font);
    cbmax.setPosition(800, 120);
    cbmin.setPosition(800, 400);
    cbmax.setCharacterSize(20);
    cbmin.setCharacterSize(20);

    first_mark .setPosition(260, 950);
    second_mark.setPosition(770, 950);
    first_mark .setCharacterSize(40);
    second_mark.setCharacterSize(40);
    first_mark .setFillColor(sf::Color::Green);    
    second_mark.setFillColor(sf::Color::Green);

    sf::VertexBuffer triangles(sf::TrianglesStrip);
    triangles.create(512);
    triangles.update(vertices);
    tex.setSmooth(true);
    /*std::thread fred([&window, &grid](){
        while(window.isOpen()){
            double sigma;
            std::cin >> sigma;
            grid.sigma.sigma_metal = sigma;
        }
    });*/
    bool isopen = true;
    while (isopen){
        sf::Event event;
        while (window.pollEvent(event)){
            if (event.type == sf::Event::Closed){
                window.close();std::abort();
            }
            if(event.type == sf::Event::KeyPressed){
                if(event.key.code == sf::Keyboard::Escape){
                    window.close();
                    std::abort();
                }
            }
            if(event.type == sf::Event::MouseButtonPressed){
                for(auto& b : buttons){
                    b.on_screen_click(window);
                }
            }
        }
        constexpr double timestep = 2e-10;
        auto t1 = nanoTime();
        for(size_t c = 0;c < 10;c++){
            grid.B.block(grid.m / 4, 0, grid.m / 2, 1) = bcond(time) * 5;
            if(!paused){
                grid_with_positions g2 = grid;
                g2.B += grid.interior_laplacian() * timestep;
                grid.B += (grid.interior_laplacian() + g2.interior_laplacian()) * 0.5 * timestep;
                time += timestep;
            }
            maxf = std::max(maxf, std::abs(grid.B(16, 120)));
            if(time > 5e5){
                window.close();
                std::cout << thickness * 1000 << " " << maxf << std::endl;
                isopen = false;
                break;
            }
        }
        auto t2 = nanoTime();
        //std::cout << "Speed: " << 1.0 / (t2 - t1) << std::endl;
        window.clear();
        std::vector<unsigned char> texdata((grid.m) * (grid.n) * 4);
        for(size_t i = 0;i < grid.m;i++){
            for(size_t j = 0;j < grid.n;j++){
                texdata[(i * (grid.n) + j) * 4]     = fcolor(turbo_cm[fcolor(std::abs(grid.B(i, j) / cbmaxv))][0]);
                texdata[(i * (grid.n) + j) * 4 + 1] = fcolor(turbo_cm[fcolor(std::abs(grid.B(i, j) / cbmaxv))][1]);
                texdata[(i * (grid.n) + j) * 4 + 2] = fcolor(turbo_cm[fcolor(std::abs(grid.B(i, j) / cbmaxv))][2]);
                if(j > 0){
                    if(grid.positions(i, j - 1).y() < 1 && grid.positions(i, j).y() >= 1){
                        texdata[(i * (grid.n) + j) * 4 + 0] = inc_by(texdata[(i * (grid.n) + j) * 4 + 0], 20);
                        texdata[(i * (grid.n) + j) * 4 + 1] = inc_by(texdata[(i * (grid.n) + j) * 4 + 1], 20);
                        texdata[(i * (grid.n) + j) * 4 + 2] = inc_by(texdata[(i * (grid.n) + j) * 4 + 2], 20);
                    }
                    if(grid.positions(i, j - 1).y() < (1 + grid.sigma.width) && grid.positions(i, j).y() >= (1 + grid.sigma.width)){
                        texdata[(i * (grid.n) + j) * 4 + 0] = inc_by(texdata[(i * (grid.n) + j) * 4 + 0], 20);
                        texdata[(i * (grid.n) + j) * 4 + 1] = inc_by(texdata[(i * (grid.n) + j) * 4 + 1], 20);
                        texdata[(i * (grid.n) + j) * 4 + 2] = inc_by(texdata[(i * (grid.n) + j) * 4 + 2], 20);
                    }
                }
                
                texdata[(i * (grid.n) + j) * 4 + 3] = 255;
            }
        }
        timet.setString(tu_string(time * 1000) + " ms");
        tex.update(texdata.data());
        window.draw(schprit);
        window.draw(triangles);
        window.draw(cbmax);
        window.draw(cbmin);
        window.draw(first_mark );
        window.draw(second_mark);
        window.draw(timet);
        window.draw(triangles);
        counter++;
        if((counter % 1024) == 0)
            window.capture().saveToFile("frame" + tu_string(counter) + ".png");
        for(auto& b : buttons){
            b.draw(window);
        }
        window.display();
    }
    //fred.join();
}
int main(){
    scalar thickness = 0.003;
    run_with_thickness(thickness);
    for(size_t i = 0;i < 30;i++){
        if(i >= 19)
            run_with_thickness(thickness);
        thickness *= 1.1;
    }
}