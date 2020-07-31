package parts.wisdom.simplednn

fun main(args: Array<String>)
{
    val trainingData = ExampleSet(readTrainingData())
    val testData = ExampleSet(readTestData())
    check(trainingData.shape == testData.shape)

    val outputShape = Shape(1, 10)
    val classifier = Softmax(FullyConnected(trainingData.shape, outputShape), Shape(1, 10))
    println("Untrained performance: ${evaluate(classifier, testData)}")
    train(classifier, trainingData, testData)
}
