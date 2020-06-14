package parts.wisdom.simplednn

fun main(args: Array<String>)
{
    val trainingData = ExampleSet(readTrainingData())
    val testData = ExampleSet(readTestData())
    check(trainingData.shape == testData.shape)

    val classifier = FullyConnectedSoftmax(trainingData.shape, Shape(1, 10))
    println("Untrained performance: ${evaluate(classifier, testData)}")
    train(classifier, trainingData, testData)
}
