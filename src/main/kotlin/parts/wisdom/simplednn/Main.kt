package parts.wisdom.simplednn

fun main() {
    val trainingData = ExampleSet(readTrainingData())
    val testData = ExampleSet(readTestData())
    check(trainingData.shape == testData.shape)

    val outputShape = Shape(1, 10)
    val classifier = Softmax(
        FullyConnected(InputLayer(trainingData.shape), outputShape),
        Shape(1, 10)
    )
    println("Untrained performance: ${evaluate(classifier, testData)}")
    train(classifier, trainingData, testData)
}
