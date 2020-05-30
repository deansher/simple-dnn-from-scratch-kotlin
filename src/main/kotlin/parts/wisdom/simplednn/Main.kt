package parts.wisdom.simplednn

fun main(args: Array<String>)
{
    val trainingData = ExampleSet(readTrainingData())
    val exampleDims = trainingData.dims
    val testData = ExampleSet(readTestData())
    check(trainingData.dims == testData.dims)

    val dnn = SimpleClassifier(exampleDims, 10)
    // train(dnn, trainingData)
    val metrics = evaluate(dnn, testData)
    println("Final performance: $metrics")
}
