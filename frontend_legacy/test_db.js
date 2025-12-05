// frontend/test_db.js

// Use import instead of require
import dotenv from 'dotenv';
import pg from 'pg'; // Import the entire pg module

dotenv.config({ path: '.env.local' });
const { Client } = pg; // Destructure Client from pg

async function testDbConnection() {
  const connectionString = process.env.POSTGRES_URL;
  console.log('Attempting to connect to:', connectionString);

  const client = new Client({
    connectionString: connectionString,
  });

  try {
    await client.connect();
    console.log('Successfully connected to PostgreSQL!');

    // Try to create a dummy table to see if we have write access
    await client.query(`
      CREATE TABLE IF NOT EXISTS test_users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL
      );
    `);
    console.log('Successfully created/checked test_users table!');

    // Try to insert a dummy user
    await client.query(`INSERT INTO test_users (name) VALUES ('Test User');`);
    console.log('Successfully inserted a test user!');

  } catch (error) {
    console.error('Database connection or query failed:', error);
  } finally {
    await client.end();
  }
}

testDbConnection();