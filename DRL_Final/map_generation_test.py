from DRL_Final.map_generator import *

def LHR2():
    map_generator = MapGenerator("LHR2.xml", 8, 8)
    map_generator.set_range(0, 0, 0)
    map_generator.set_range(1, 1, 0)
    map_generator.set_light(0, 2, 0)
    map_generator.set_heavy(1, 2, 0)
    map_generator.set_light(2, 0, 0)
    map_generator.set_heavy(2, 1, 0)
    
    map_generator.set_range(7, 7, 1)
    map_generator.set_range(6, 6, 1)
    map_generator.set_light(7, 5, 1)
    map_generator.set_heavy(6, 5, 1)
    map_generator.set_light(5, 7, 1)
    map_generator.set_heavy(5, 6, 1)

    map_generator.generate()

if __name__ == "__main__":
    LHR2()
    V = MapVisualizer("custom\\LHR2.xml")
    V.visualize()