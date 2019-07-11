def lbf2N(lbf=1):
	""" Convert pound-force to newtons """
	N = lbf/0.22481
	return N

def inch2m(inches=1):
	""" Convert inches to meters """
	m = inches/39.370
	return m

if __name__ == "__main__":
	print "1 lbf = {:.3f} N".format(lbf2N())
	print "1 in = {:.3f} m".format(inch2m())
	print "1 lbf/in = {:.5f} N/m".format(lbf2N()/inch2m())

