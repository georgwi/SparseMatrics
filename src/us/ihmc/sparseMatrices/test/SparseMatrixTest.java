package us.ihmc.sparseMatrices.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.ops.CommonOps;
import org.junit.Test;

import us.ihmc.sparseMatrices.SparseMatrix;
import us.ihmc.sparseMatrices.SparseMatrixInverter;

public class SparseMatrixTest
{
   @Test
   public void testSparseMatrix()
   {
      Random random = new Random(492911L);
      int numColumns = random.nextInt(50);
      int numRows = random.nextInt(50);

      DenseMatrix64F reference = new DenseMatrix64F(numRows, numColumns);
      SparseMatrix matrix = new SparseMatrix(numRows, numColumns);

      assertEquals(reference.getNumRows(), matrix.getRows());
      assertEquals(reference.getNumCols(), matrix.getColumns());

      for (int i = 0; i < 50; i++)
      {
         int row = random.nextInt(numRows);
         int column = random.nextInt(numColumns);
         double value = random.nextDouble();
         reference.set(row, column, value);
         matrix.set(row, column, value);
      }

      for (int row = 0; row < numRows; row++)
      {
         for (int column = 0; column < numColumns; column++)
         {
            assertEquals(reference.get(row, column), matrix.get(row, column), 1.0E-20);
         }
      }
   }

   @Test
   public void checkDimensions()
   {
      Random random = new Random(52911L);
      SparseMatrix matrix = new SparseMatrix();
      for (int i = 0; i < 100; i++)
      {
         int row = random.nextInt(20) - 10;
         int column = random.nextInt(20) - 10;

         try
         {
            matrix.get(row, column);
            fail("Expected Exception");
         }
         catch (Exception e)
         {
         }

         try
         {
            matrix.set(row, column, 0.0);
            fail("Expected Exception");
         }
         catch (Exception e)
         {
         }
      }
   }

   @Test
   public void testMultiplication()
   {
      Random random = new Random(492911L);

      int maxMatrixSize = 100;
      int maxEntriesInMatrix = 100;
      int iterations = 100;

      long startTime;
      long totalTimeDense = 0;
      long totalTimeSparse = 0;

      for (int i = 0; i < iterations; i++)
      {
         // Create matrices with random dimensions.
         // Matrix A is nxm and B is mxp
         int n = random.nextInt(maxMatrixSize) + 1;
         int m = random.nextInt(maxMatrixSize) + 1;
         int p = random.nextInt(maxMatrixSize) + 1;

         DenseMatrix64F referenceA = new DenseMatrix64F(n, m);
         DenseMatrix64F referenceB = new DenseMatrix64F(m, p);
         SparseMatrix matrixA = new SparseMatrix(n, m);
         SparseMatrix matrixB = new SparseMatrix(m, p);

         // set random entries
         for (int j = 0; j < maxEntriesInMatrix; j++)
         {
            int row = random.nextInt(n);
            int column = random.nextInt(m);
            double value = random.nextDouble();
            referenceA.set(row, column, value);
            matrixA.set(row, column, value);
         }
         for (int j = 0; j < maxEntriesInMatrix; j++)
         {
            int row = random.nextInt(m);
            int column = random.nextInt(p);
            double value = random.nextDouble();
            referenceB.set(row, column, value);
            matrixB.set(row, column, value);
         }

         DenseMatrix64F referenceResult = new DenseMatrix64F(n, p);
         SparseMatrix result = new SparseMatrix();

         startTime = System.nanoTime();
         CommonOps.mult(referenceA, referenceB, referenceResult);
         totalTimeDense += System.nanoTime() - startTime;

         startTime = System.nanoTime();
         result.mult(matrixA, matrixB);
         totalTimeSparse += System.nanoTime() - startTime;

         assertEquals(referenceResult.getNumRows(), result.getRows());
         assertEquals(referenceResult.getNumCols(), result.getColumns());

         for (int row = 0; row < n; row++)
         {
            for (int column = 0; column < p; column++)
            {
               assertEquals(referenceResult.get(row, column), result.get(row, column), 1.0E-10);
            }
         }
      }

      double timeDenseSeconds = 1.0E-9 * totalTimeDense;
      double timeSparseSeconds = 1.0E-9 * totalTimeSparse;
      System.out.println("Average time dense: " + timeDenseSeconds / iterations);
      System.out.println("Average time sparse: " + timeSparseSeconds / iterations);
   }

   @Test
   public void testInversionWithFullMatricesAgainstDense()
   {
      Random random = new Random(492911L);

      int matrixSize = 20;
      int maxEntries = 1000;
      int iterations = 1000;

      long startTime;
      long totalTimeDense = 0;
      long totalTimeSparse = 0;

      SparseMatrixInverter inverter = new SparseMatrixInverter();
      LinearSolver<DenseMatrix64F> referenceSolver = LinearSolverFactory.linear(0);

      int sparseInversions = 0;
      int denseInversions = 0;

      for (int i = 0; i < iterations; i++)
      {
         // Create square matrix with random dimensions.
         // Matrix is nxn
         int n = matrixSize;

         DenseMatrix64F reference = new DenseMatrix64F(n, n);
         SparseMatrix matrix = new SparseMatrix(n, n);

         // set random entries
         List<Integer> indeces = new ArrayList<>();
         for (int j = 0; j < n; j++)
         {
            indeces.add(new Integer(j));
         }
         Collections.shuffle(indeces, random);
         for (int j = 0; j < n; j++)
         {
            double value = random.nextDouble();
            reference.set(j, indeces.get(j), value);
            matrix.set(j, indeces.get(j), value);
         }

         for (int j = 0; j < maxEntries - n; j++)
         {
            int row = random.nextInt(n);
            int column = random.nextInt(n);
            double value = random.nextDouble();
            reference.set(row, column, value);
            matrix.set(row, column, value);
         }

         SparseMatrix inverse = new SparseMatrix();
         DenseMatrix64F indentity = new DenseMatrix64F(n, n);
         CommonOps.setIdentity(indentity);
         DenseMatrix64F referenceInverse = new DenseMatrix64F(n, n);

         startTime = System.nanoTime();
         boolean possible = referenceSolver.setA(reference);
         if (possible)
         {
            referenceSolver.solve(indentity, referenceInverse);
         }
         long denseTime =  System.nanoTime() - startTime;

         startTime = System.nanoTime();
         boolean nonSingular = inverter.invert(matrix, inverse);
         long sparseTime =  System.nanoTime() - startTime;

         if (possible != nonSingular)
         {
            System.out.println("Reference solver: " + (possible ? "invertible" : "singular"));
            System.out.println("Sparse solver: " + (nonSingular ? "invertible" : "singular"));

            if (possible)
            {
               fail("The solvers disagree as to whether the matrix is singular.");
            }
         }

         if (possible)
         {
            denseInversions++;
            totalTimeDense += denseTime;
         }
         if (nonSingular)
         {
            sparseInversions++;
            totalTimeSparse += sparseTime;
         }

         if (possible && nonSingular)
         {
            for (int row = 0; row < n; row++)
            {
               for (int column = 0; column < n; column++)
               {
                  if (Math.abs(referenceInverse.get(row, column)) < 0.1)
                  {
                     assertEquals(referenceInverse.get(row, column), inverse.get(row, column), 1.0E-5);
                  }
                  else
                  {
                     double error = Math.abs(1.0 - referenceInverse.get(row, column) / inverse.get(row, column));
                     assertTrue("(" + referenceInverse.get(row, column) + ") Error too big: " + error, error < 1.0E-3);
                  }
               }
            }
         }
      }

      double timeDenseSeconds = 1.0E-9 * totalTimeDense;
      double timeSparseSeconds = 1.0E-9 * totalTimeSparse;
      System.out.println("Average time dense: " + timeDenseSeconds / denseInversions);
      System.out.println("Average time sparse: " + timeSparseSeconds / sparseInversions);
   }

   @Test
   public void testInversionOfLargeEmptyMatrices()
   {
      Random random = new Random(492911L);

      int matrixSize = 1000;
      int iterations = 10;

      SparseMatrixInverter inverter = new SparseMatrixInverter();

      for (int i = 0; i < iterations; i++)
      {
         // Create square matrix with random dimensions.
         // Matrix is nxn
         int n = matrixSize;

         SparseMatrix matrix = new SparseMatrix(n, n);

         // set random entries
         List<Integer> indeces = new ArrayList<>();
         for (int j = 0; j < n; j++)
         {
            indeces.add(new Integer(j));
         }
         Collections.shuffle(indeces, random);
         for (int j = 0; j < n; j++)
         {
            double value = random.nextDouble();
            matrix.set(j, indeces.get(j), value);
         }

         SparseMatrix inverse = new SparseMatrix();
         if (!inverter.invert(matrix, inverse))
         {
            System.out.println("Encountered singular matrix.");
            continue;
         }

         SparseMatrix eye = new SparseMatrix();
         eye.mult(inverse, matrix);

         for (int row = 0; row < n; row++)
         {
            for (int column = 0; column < n; column++)
            {
               if (row == column)
               {
                  assertEquals(1.0, eye.get(row, column), 1.0E-10);
               }
               else
               {
                  assertFalse(eye.contains(row, column));
               }
            }
         }
      }
   }

   @Test
   public void testInversionOfLargeBlockMatrices()
   {
      Random random = new Random(492911L);

      int matrixSize = 100;
      int iterations = 500;
      int blocks = 5;

      SparseMatrixInverter inverter = new SparseMatrixInverter();
      LinearSolver<DenseMatrix64F> referenceSolver = LinearSolverFactory.linear(0);

      long startTime;
      long totalTimeDense = 0;
      long totalTimeSparse = 0;

      DenseMatrix64F reference = new DenseMatrix64F(matrixSize, matrixSize);
      SparseMatrix matrix = new SparseMatrix(matrixSize, matrixSize);

      SparseMatrix sparseInverse = new SparseMatrix();
      DenseMatrix64F denseInverse = new DenseMatrix64F(matrixSize, matrixSize);
      DenseMatrix64F indentity = new DenseMatrix64F(matrixSize, matrixSize);
      CommonOps.setIdentity(indentity);

      for (int i = 0; i < iterations; i++)
      {
         matrix.clear();
         CommonOps.fill(reference, 0.0);

         // make three blocks:
         int blockSize = matrixSize / blocks;
         for (int row = 0; row < blockSize; row++)
         {
            for (int column = 0; column < blockSize; column++)
            {
               for (int block = 0; block < blocks; block++)
               {
                  int offset = block * blockSize;
                  double value = random.nextDouble();
                  reference.set(row + offset, column + offset, value);
                  matrix.set(row + offset, column + offset, value);
               }
            }
         }

         startTime = System.nanoTime();
         if (!inverter.invert(matrix, sparseInverse))
         {
            fail("Sparse solver encountered singular matrix.");
         }
         totalTimeSparse += System.nanoTime() - startTime;

         startTime = System.nanoTime();
         if (!referenceSolver.setA(reference))
         {
            fail("Reference solver encountered singular matrix.");
         }
         referenceSolver.solve(indentity, denseInverse);
         totalTimeDense += System.nanoTime() - startTime;
      }

      double timeDenseSeconds = 1.0E-9 * totalTimeDense;
      double timeSparseSeconds = 1.0E-9 * totalTimeSparse;
      System.out.println("Average time dense inversion: " + timeDenseSeconds / iterations);
      System.out.println("Average time sparse inversion: " + timeSparseSeconds / iterations);
   }
}
