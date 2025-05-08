import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService
  extends PrismaClient
  implements OnModuleInit, OnModuleDestroy
{
  private static instance: PrismaService;

  constructor() {
    super({
      // Add connection pooling configuration
      datasources: {
        db: {
          url: process.env.DATABASE_URL,
        },
      },
      // Configure connection pool
      log: ['error', 'warn'],
    });

    if (!PrismaService.instance) {
      PrismaService.instance = this;
    }

    return PrismaService.instance;
  }

  async onModuleInit() {
    await this.$connect();
  }

  async onModuleDestroy() {
    await this.$disconnect();
  }
}
