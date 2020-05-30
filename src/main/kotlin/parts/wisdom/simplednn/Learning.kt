package parts.wisdom.simplednn

import koma.matrix.Matrix
import koma.matrix.MatrixTypes
import koma.zeros

private const val BATCH_SIZE = 50
private const val NUM_EPOCHS = 20

data class Example(val label: Int, val matrix: Matrix<Double>) {
    val dims = ExampleDims(matrix.numRows(), matrix.numCols())
}

data class ExampleDims(
    val numRows: Int,
    val numCols: Int
) {
    fun requireSameSize(x: Example) {
        require(x.matrix.numRows() == numRows) {
            "example has ${x.matrix.numRows()}; should have $numRows"
        }
        require(x.matrix.numCols() == numCols) {
            "example has ${x.matrix.numCols()}; should have $numCols"
        }
    }
}

fun requireConsistentDims(xs: List<Example>): ExampleDims {
    require(xs.isNotEmpty()) { "must have at least one example" }
    val dims = xs[0].dims
    for (x in xs) {
        dims.requireSameSize(x)
    }
    return dims
}

data class ExampleSet(val examples: List<Example>) {
    val dims = requireConsistentDims(examples)
}

fun pickBatches(examples: ExampleSet, batchSize: Int): List<List<Example>> =
    examples.examples.shuffled().chunked(batchSize)

data class EvaluationMetrics(val accuracy: Float) {

}

//fun train(classifier: SimpleClassifier, trainingData: ExampleSet) {
//    val batches = pickBatches(trainingData, BATCH_SIZE)
//    for (epoch in 1..NUM_EPOCHS) {
//        val deltaParams = zeros(classifier.paramNumRows, classifier.paramNumCols, MatrixTypes.FloatType)
//        println("After epoch $epoch, ${TODO()}")
//    }
//}

fun evaluate(classifier: SimpleClassifier, testData: ExampleSet): EvaluationMetrics {
    var numRight = 0
    var numWrong = 0
    for (x in testData.examples) {
        if (classifier.inferClass(x) == x.label) {
            numRight++
        } else {
            numWrong++
        }
    }
    return EvaluationMetrics(numRight.toFloat() / (numRight + numWrong).toFloat())
}


