package parts.wisdom.simplednn

import koma.matrix.Matrix

private const val BATCH_SIZE = 20
private const val NUM_EPOCHS = 20

data class Coords(val row: Int, val col: Int) {
    constructor(idx: IntArray) :
            this(
                idx.let {
                    require(it.size == 2) { "Can't compare to an index of length ${it.size}" }
                    it[0]
                },
                idx[1]
            )

    fun equalsIdx(idx: IntArray): Boolean =
        idx.size == 2 && idx[0] == row && idx[1] == col
}

/**
 * A training example. The label is a combination of row and column because we use two dimensions for all outputs.
 */
data class Example(
    val label: Coords,
    val matrix: Matrix<Double>
) {
    val shape = Shape(matrix.numRows(), matrix.numCols())
}

data class Shape(
    val numRows: Int,
    val numCols: Int
) {
    fun requireSameShape(x: Example) {
        require(x.matrix.numRows() == numRows) {
            "example has ${x.matrix.numRows()}; should have $numRows"
        }
        require(x.matrix.numCols() == numCols) {
            "example has ${x.matrix.numCols()}; should have $numCols"
        }
    }
}

fun requireAllSameShape(xs: List<Example>): Shape {
    require(xs.isNotEmpty()) { "must have at least one example" }
    val shape = xs[0].shape
    for (x in xs) {
        shape.requireSameShape(x)
    }
    return shape
}

data class ExampleSet(val examples: List<Example>) {
    val shape = requireAllSameShape(examples)
}

fun pickBatches(examples: ExampleSet, batchSize: Int): List<List<Example>> =
    examples.examples.shuffled().chunked(batchSize)

data class EvaluationMetrics(val accuracy: Float)

fun train(
    classifier: FullyConnectedSoftmax,
    trainingData: ExampleSet,
    testData: ExampleSet
) {
    for (epoch in 1..NUM_EPOCHS) {
        for (batch in pickBatches(trainingData, BATCH_SIZE)) {
            classifier.trainBatch(batch)
        }
        val metrics = evaluate(classifier, testData)
        println("After epoch $epoch, $metrics")
    }
}

fun evaluate(classifier: FullyConnectedSoftmax, testData: ExampleSet): EvaluationMetrics {
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


