import tensorflow as tf

a= tf.Variable(23)
b= tf.constant(234)

print("the Addition of a and b is :",a+b)
print("the multiplication of a and b is :",a*b)

x=int(input("enter a value for a:"))
b= tf.Variable(x)

print(a-b)