# Genetic-Algorithms
Some simple projects using genetic algorithms

# String Reconstruction
Takes a target string, and reconstructs it from random characters
Example execution:

reproduce_string("According to all known laws of aviation, there is no way a bee should be able to fly.", generation_size=200, generations=1000, elitism_ratio=0.2, mutation_rate=0.01)

Generation: 0/1000
Completed in: 0.002
Best String: Aa;DV]zA^sF^5nSGGmn@B>.U1E{d)=*5AQI2>{YbUtJO|F-+f$n#{q1_<Z>el7}!,fwidxHwn^u4ETA7OB8Qn
Fitness: 0.06


Generation: 100/1000
Best String: AcRordin@Xtog>0lcRnoBn laws )fh&Aiation, tJOrF-is no way g eee}<h)+id[be\5bletao fly.
Fitness: 0.66


Generation: 200/1000
Best String: AccordingXto all known laws of aviation, therH-is no way g eee <h)uid[be\5ble to fly.
Fitness: 0.87


Generation: 300/1000
Best String: According to all known laws of aviation, there is no way a bee sh)uld be\able to fly.
Fitness: 0.98


Generation: 400/1000
Best String: According to all known laws of aviation, there is no way a bee sh)uld be\able to fly.
Fitness: 0.98


Generation: 500/1000
Best String: According to all known laws of aviation, there is no way a bee sh)uld be able to fly.
Fitness: 0.99


Target String: According to all known laws of aviation, there is no way a bee should be able to fly.
Reconstructed String: According to all known laws of aviation, there is no way a bee should be able to fly.
Completed in 576 generations


# Image Reconstruction
Takes a target image, and reconstructs it from random noise


Concept from: https://github.com/efefurkankarakaya/reproducer
