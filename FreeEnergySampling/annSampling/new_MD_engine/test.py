#!/usr/bin/env python3

class test(object):
  def a(self):
    return 1
  def b(self):
    print(__class__.__name__)

test().b()
