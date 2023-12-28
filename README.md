<p align="center">
  <img width="250" height="250" src="assets/tone_arm.png">
<br>
Vinyl record playback noise from physical simulations.
</p>

------

Features
--------

* Hiss and popping noise due to dust and damage
* High frequency roll-off due to stylus size
* Wow and flutter low frequency distortion
* Distortion at high amplitudes from cartridge non-linearity
* RIAA playback equalisation

Installation
------------

Install from PyPI
```bash
pip install tonearm
```

or install from the Github repository
```bash
git clone git@github.com:dpohanlon/ToneArm.git
cd ToneArm
pip install .
```

Usage
---

To add noise and distortion to a `.wav` file
```bash
tonearm --input filename.wav --name output
```

or to generate a sample of just the noise, call with no file name and the length of the sample to generate
```bash
tonearm --name noise --length 10
```
