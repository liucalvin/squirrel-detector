from pi5RC import pi5RC
from time import sleep

pwm0 = pi5RC(12)
pwm1 = pi5RC(13)


pwm0.set(0)
sleep(1)
pwm0.set(500)
sleep(1)
pwm0.set(1000)
