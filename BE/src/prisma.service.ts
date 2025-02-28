import { Injectable, OnModuleDestroy, OnModuleInit } from '@nestjs/common';
import { PrismaClient } from '@prisma/client';

@Injectable()
export class PrismaService
  extends PrismaClient
  implements OnModuleInit, OnModuleDestroy
{
  async onModuleInit() {
    console.log('Connect to database successfully');
    await this.$connect();
  }

  async onModuleDestroy() {
    console.log('Disconnect to MYSQL successfully');
    await this.$disconnect();
  }
}
