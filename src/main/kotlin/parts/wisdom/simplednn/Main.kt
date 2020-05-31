package parts.wisdom.simplednn

fun main(args: Array<String>)
{
    val trainingData = ExampleSet(readTrainingData())
    val exampleDims = trainingData.dims
    val testData = ExampleSet(readTestData())
    check(trainingData.dims == testData.dims)

    val classifier = SimpleClassifier(exampleDims, 10)
    println("Untrained performance: ${evaluate(classifier, testData)}")
    train(classifier, trainingData, testData)
}
